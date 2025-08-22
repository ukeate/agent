"""
重构后的分布式事件处理系统
将复杂方法拆分为更小、更专注的方法以提高可维护性
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
import structlog
try:
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis
    except ImportError:
        aioredis = None

logger = structlog.get_logger(__name__)


@dataclass
class DistributedEvent:
    """分布式事件"""
    event_id: str
    event_type: str
    source_node: str
    target_nodes: List[str]
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


class EventSerializer:
    """事件序列化器"""
    
    @staticmethod
    def serialize(event: DistributedEvent) -> Dict[str, Any]:
        """序列化事件"""
        return {
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
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> DistributedEvent:
        """反序列化事件"""
        return DistributedEvent(
            event_id=data["event_id"],
            event_type=data["event_type"],
            source_node=data["source_node"],
            target_nodes=data["target_nodes"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 0),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


class EventPublisher:
    """事件发布器"""
    
    def __init__(self, redis_client, event_stream_prefix: str = "events:", max_stream_length: int = 10000):
        self.redis_client = redis_client
        self.event_stream_prefix = event_stream_prefix
        self.max_stream_length = max_stream_length
    
    async def publish_to_stream(self, event: DistributedEvent) -> bool:
        """发布事件到Redis流"""
        try:
            event_data = EventSerializer.serialize(event)
            stream_key = f"{self.event_stream_prefix}{event.event_type}"
            
            await self.redis_client.xadd(
                stream_key,
                event_data,
                maxlen=self.max_stream_length
            )
            
            logger.debug("事件发布到流成功", event_id=event.event_id, stream_key=stream_key)
            return True
            
        except Exception as e:
            logger.error("发布事件到流失败", event_id=event.event_id, error=str(e))
            return False
    
    async def notify_target_nodes(self, event: DistributedEvent, stream_key: str) -> bool:
        """通知目标节点"""
        try:
            notification_data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "stream_key": stream_key
            }
            
            # 并行发送通知
            notification_tasks = []
            for target_node in event.target_nodes:
                task = self._send_single_notification(target_node, notification_data)
                notification_tasks.append(task)
            
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            
            logger.debug(
                "节点通知完成",
                event_id=event.event_id,
                total_targets=len(event.target_nodes),
                success_count=success_count
            )
            
            return success_count > 0
            
        except Exception as e:
            logger.error("通知目标节点失败", event_id=event.event_id, error=str(e))
            return False
    
    async def _send_single_notification(self, target_node: str, notification_data: Dict[str, Any]) -> bool:
        """发送单个节点通知"""
        try:
            notification_key = f"node_notifications:{target_node}"
            await self.redis_client.lpush(
                notification_key,
                json.dumps(notification_data)
            )
            await self.redis_client.expire(notification_key, 3600)
            return True
        except Exception as e:
            logger.warning("发送通知失败", target_node=target_node, error=str(e))
            return False


class EventConsumer:
    """事件消费器"""
    
    def __init__(self, redis_client, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        self.running = False
    
    async def start_listening(self):
        """开始监听事件"""
        self.running = True
        await self._listen_for_notifications()
    
    async def stop_listening(self):
        """停止监听事件"""
        self.running = False
    
    async def _listen_for_notifications(self):
        """监听事件通知 - 重构后的方法"""
        notification_key = f"node_notifications:{self.node_id}"
        
        while self.running:
            try:
                result = await self._wait_for_notification(notification_key)
                
                if result:
                    notification = self._parse_notification(result)
                    if notification:
                        await self._handle_notification(notification)
                        
            except Exception as e:
                logger.error("监听通知失败", error=str(e))
                await self._handle_listening_error()
    
    async def _wait_for_notification(self, notification_key: str) -> Optional[bytes]:
        """等待通知"""
        try:
            result = await self.redis_client.brpop(notification_key, timeout=1)
            return result[1] if result else None
        except Exception as e:
            logger.debug("等待通知超时或失败", error=str(e))
            return None
    
    def _parse_notification(self, notification_data: bytes) -> Optional[Dict[str, Any]]:
        """解析通知数据"""
        try:
            return json.loads(notification_data.decode())
        except Exception as e:
            logger.warning("解析通知失败", error=str(e))
            return None
    
    async def _handle_notification(self, notification: Dict[str, Any]):
        """处理事件通知 - 重构后的方法"""
        try:
            event = await self._fetch_event_from_stream(notification)
            if event:
                await self._process_event(event)
        except Exception as e:
            logger.error("处理通知失败", notification=notification, error=str(e))
    
    async def _fetch_event_from_stream(self, notification: Dict[str, Any]) -> Optional[DistributedEvent]:
        """从流中获取事件"""
        try:
            event_id = notification["event_id"]
            stream_key = notification["stream_key"]
            
            # 从流中读取事件
            events = await self.redis_client.xread({stream_key: "$"}, count=1, block=1000)
            
            for stream, messages in events:
                for message_id, fields in messages:
                    if fields.get("event_id") == event_id:
                        return EventSerializer.deserialize(fields)
            
            return None
            
        except Exception as e:
            logger.error("从流获取事件失败", notification=notification, error=str(e))
            return None
    
    async def _process_event(self, event: DistributedEvent):
        """处理事件 - 抽象方法，由子类实现"""
        logger.info("处理事件", event_id=event.event_id, event_type=event.event_type)
    
    async def _handle_listening_error(self):
        """处理监听错误"""
        await asyncio.sleep(1)


class LoadBalancingStrategy:
    """负载均衡策略"""
    
    @staticmethod
    def calculate_average_load(nodes: Dict[str, Any]) -> float:
        """计算平均负载"""
        if not nodes:
            return 0.0
        
        total_load = sum(node.get('load', 0) for node in nodes.values())
        return total_load / len(nodes)
    
    @staticmethod
    def identify_high_load_nodes(nodes: Dict[str, Any], threshold_factor: float = 1.2) -> List[str]:
        """识别高负载节点"""
        avg_load = LoadBalancingStrategy.calculate_average_load(nodes)
        threshold = avg_load * threshold_factor
        
        return [
            node_id for node_id, node in nodes.items()
            if node.get('load', 0) > threshold
        ]
    
    @staticmethod
    def identify_low_load_nodes(nodes: Dict[str, Any], threshold_factor: float = 0.8) -> List[str]:
        """识别低负载节点"""
        avg_load = LoadBalancingStrategy.calculate_average_load(nodes)
        threshold = avg_load * threshold_factor
        
        return [
            node_id for node_id, node in nodes.items()
            if node.get('load', 0) < threshold
        ]


class LoadBalancer:
    """负载均衡器 - 重构后的版本"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.strategy = LoadBalancingStrategy()
    
    async def rebalance_load(self, nodes: Dict[str, Any], current_role: str) -> Dict[str, Any]:
        """重新平衡负载 - 重构后的方法"""
        if current_role != "leader":
            return {"error": "只有领导者可以重新平衡负载"}
        
        # 分析负载状况
        load_analysis = self._analyze_load_distribution(nodes)
        
        # 生成重平衡计划
        rebalancing_plan = self._create_rebalancing_plan(load_analysis)
        
        # 执行重平衡
        execution_result = await self._execute_rebalancing(rebalancing_plan)
        
        return {
            "analysis": load_analysis,
            "plan": rebalancing_plan,
            "execution": execution_result
        }
    
    def _analyze_load_distribution(self, nodes: Dict[str, Any]) -> Dict[str, Any]:
        """分析负载分布"""
        avg_load = self.strategy.calculate_average_load(nodes)
        high_load_nodes = self.strategy.identify_high_load_nodes(nodes)
        low_load_nodes = self.strategy.identify_low_load_nodes(nodes)
        
        return {
            "average_load": avg_load,
            "high_load_nodes": high_load_nodes,
            "low_load_nodes": low_load_nodes,
            "total_nodes": len(nodes),
            "needs_rebalancing": len(high_load_nodes) > 0 and len(low_load_nodes) > 0
        }
    
    def _create_rebalancing_plan(self, load_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建重平衡计划"""
        if not load_analysis["needs_rebalancing"]:
            return {"actions": [], "reason": "无需重平衡"}
        
        actions = []
        high_load_nodes = load_analysis["high_load_nodes"]
        low_load_nodes = load_analysis["low_load_nodes"]
        
        # 简化的重平衡策略：配对高低负载节点
        for i, high_node in enumerate(high_load_nodes):
            if i < len(low_load_nodes):
                low_node = low_load_nodes[i]
                actions.append({
                    "type": "migrate_tasks",
                    "from_node": high_node,
                    "to_node": low_node,
                    "task_count": "auto"  # 自动计算迁移数量
                })
        
        return {
            "actions": actions,
            "estimated_improvement": len(actions) * 0.1  # 估算改善程度
        }
    
    async def _execute_rebalancing(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行重平衡计划"""
        if not plan["actions"]:
            return {"status": "skipped", "reason": plan.get("reason", "无操作")}
        
        executed_actions = []
        failed_actions = []
        
        for action in plan["actions"]:
            try:
                # 这里应该调用实际的任务迁移逻辑
                result = await self._execute_single_action(action)
                if result["success"]:
                    executed_actions.append(action)
                else:
                    failed_actions.append({"action": action, "error": result["error"]})
            except Exception as e:
                failed_actions.append({"action": action, "error": str(e)})
        
        return {
            "status": "completed" if not failed_actions else "partial",
            "executed_actions": len(executed_actions),
            "failed_actions": len(failed_actions),
            "failures": failed_actions
        }
    
    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个重平衡动作"""
        # 占位符实现 - 实际应该包含真正的任务迁移逻辑
        logger.info("执行负载重平衡动作", action=action)
        
        # 模拟执行结果
        return {
            "success": True,
            "transferred_tasks": 5,
            "execution_time": 1.2
        }


class EventLoopProcessor:
    """事件循环处理器 - 重构后的版本"""
    
    def __init__(self, redis_client, node_id: str, event_queue_prefix: str = "event_queue:"):
        self.redis_client = redis_client
        self.node_id = node_id
        self.event_queue_prefix = event_queue_prefix
        self.stats = {"events_received": 0, "events_processed": 0, "events_failed": 0}
        self.running = False
    
    async def start_processing(self):
        """开始处理事件循环 - 重构后的方法"""
        if not self.redis_client:
            logger.warning("Redis客户端未配置，跳过事件处理")
            return
        
        self.running = True
        queue_key = f"{self.event_queue_prefix}{self.node_id}"
        
        while self.running:
            try:
                event_data = await self._fetch_event_from_queue(queue_key)
                
                if event_data:
                    event = self._reconstruct_event(event_data)
                    if event:
                        await self._process_single_event(event)
                        
            except Exception as e:
                logger.error("事件处理循环失败", error=str(e))
                await self._handle_processing_error()
    
    async def stop_processing(self):
        """停止处理事件循环"""
        self.running = False
    
    async def _fetch_event_from_queue(self, queue_key: str) -> Optional[Dict[str, Any]]:
        """从队列获取事件"""
        try:
            result = await self.redis_client.brpop(queue_key, timeout=1)
            if result:
                _, event_data_bytes = result
                return json.loads(event_data_bytes.decode())
            return None
        except Exception as e:
            logger.debug("获取队列事件失败", error=str(e))
            return None
    
    def _reconstruct_event(self, event_data: Dict[str, Any]) -> Optional[Any]:
        """重建事件对象"""
        try:
            # 这里应该根据实际的Event类进行重建
            # 简化实现，返回字典格式
            return {
                "id": event_data.get("id"),
                "type": event_data.get("type", "unknown"),
                "source": event_data.get("source", ""),
                "target": event_data.get("target"),
                "data": event_data.get("data", {}),
                "timestamp": event_data.get("timestamp"),
                "correlation_id": event_data.get("correlation_id"),
                "conversation_id": event_data.get("conversation_id"),
                "session_id": event_data.get("session_id"),
                "priority": event_data.get("priority", "normal")
            }
        except Exception as e:
            logger.error("重建事件对象失败", event_data=event_data, error=str(e))
            return None
    
    async def _process_single_event(self, event: Dict[str, Any]):
        """处理单个事件"""
        try:
            self.stats["events_received"] += 1
            
            # 这里应该调用实际的事件处理逻辑
            await self._handle_event_business_logic(event)
            
            self.stats["events_processed"] += 1
            logger.debug("事件处理成功", event_id=event.get("id"))
            
        except Exception as e:
            self.stats["events_failed"] += 1
            logger.error("处理单个事件失败", event_id=event.get("id"), error=str(e))
    
    async def _handle_event_business_logic(self, event: Dict[str, Any]):
        """处理事件业务逻辑 - 抽象方法"""
        # 占位符 - 应该由具体实现类提供
        logger.info("处理事件业务逻辑", event_type=event.get("type"))
    
    async def _handle_processing_error(self):
        """处理处理错误"""
        await asyncio.sleep(1)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        return self.stats.copy()


# 重构后的分布式事件总线
class RefactoredDistributedEventBus:
    """重构后的分布式事件总线"""
    
    def __init__(self, redis_client, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        
        # 组件化设计
        self.event_publisher = EventPublisher(redis_client)
        self.event_consumer = EventConsumer(redis_client, node_id)
        self.load_balancer = LoadBalancer(node_id)
        self.event_processor = EventLoopProcessor(redis_client, node_id)
        
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
    
    async def start(self):
        """启动事件总线"""
        self.running = True
        
        # 启动各个组件
        await asyncio.gather(
            self.event_consumer.start_listening(),
            self.event_processor.start_processing()
        )
        
        logger.info("重构后的分布式事件总线启动完成", node_id=self.node_id)
    
    async def stop(self):
        """停止事件总线"""
        self.running = False
        
        # 停止各个组件
        await asyncio.gather(
            self.event_consumer.stop_listening(),
            self.event_processor.stop_processing()
        )
        
        logger.info("重构后的分布式事件总线已停止", node_id=self.node_id)
    
    async def publish(self, event: DistributedEvent) -> bool:
        """发布事件 - 重构后简化的方法"""
        # 发布到流
        stream_published = await self.event_publisher.publish_to_stream(event)
        
        if not stream_published:
            return False
        
        # 通知目标节点
        stream_key = f"{self.event_publisher.event_stream_prefix}{event.event_type}"
        nodes_notified = await self.event_publisher.notify_target_nodes(event, stream_key)
        
        return stream_published and nodes_notified
    
    async def subscribe(self, event_type: str, handler: Callable):
        """订阅事件类型"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def rebalance_cluster_load(self, nodes: Dict[str, Any], current_role: str) -> Dict[str, Any]:
        """重平衡集群负载"""
        return await self.load_balancer.rebalance_load(nodes, current_role)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "node_id": self.node_id,
            "running": self.running,
            "processing_stats": self.event_processor.get_processing_stats(),
            "subscribers": {event_type: len(handlers) for event_type, handlers in self.subscribers.items()}
        }