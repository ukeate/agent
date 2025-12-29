"""
消息可靠性保证实现
支持消息确认、重试机制、死信队列和消息持久化
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import uuid
import time
import json
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from .models import Message, MessageType, MessagePriority, DeliveryMode

from src.core.logging import get_logger
logger = get_logger(__name__)

class RetryPolicy(Enum):
    """重试策略"""
    NONE = auto()              # 不重试
    FIXED_INTERVAL = auto()    # 固定间隔
    EXPONENTIAL_BACKOFF = auto()  # 指数退避
    LINEAR_BACKOFF = auto()    # 线性退避

class MessageStatus(Enum):
    """消息状态"""
    PENDING = auto()           # 待处理
    SENT = auto()              # 已发送
    ACKNOWLEDGED = auto()      # 已确认
    FAILED = auto()            # 发送失败
    RETRYING = auto()          # 重试中
    DEAD_LETTER = auto()       # 进入死信队列
    EXPIRED = auto()           # 已过期

@dataclass
class RetryConfig:
    """重试配置"""
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    initial_delay: float = 1.0  # 初始延迟（秒）
    max_delay: float = 60.0     # 最大延迟（秒）
    backoff_factor: float = 2.0  # 退避因子
    jitter: bool = True         # 是否添加随机抖动

@dataclass
class ReliableMessage:
    """可靠消息包装"""
    message: Message
    message_id: str
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    require_ack: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    delivery_attempts: List[datetime] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        if self.retry_config.policy == RetryPolicy.NONE:
            return False
        # 当前状态检查：SENT状态的消息被NACK后应该能重试
        valid_statuses = [MessageStatus.PENDING, MessageStatus.FAILED, MessageStatus.RETRYING, MessageStatus.SENT]
        return self.retry_count < self.retry_config.max_retries and self.status in valid_statuses
    
    def should_retry_now(self) -> bool:
        """检查是否应该现在重试"""
        if not self.can_retry():
            return False
        # PENDING状态的消息应该立即尝试
        if self.status == MessageStatus.PENDING:
            return True
        if not self.next_retry:
            return True
        return utc_now() >= self.next_retry
    
    def calculate_next_retry_time(self) -> datetime:
        """计算下次重试时间"""
        if self.retry_config.policy == RetryPolicy.FIXED_INTERVAL:
            delay = self.retry_config.initial_delay
        elif self.retry_config.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.retry_config.initial_delay * (self.retry_count + 1)
        else:  # EXPONENTIAL_BACKOFF
            delay = min(
                self.retry_config.initial_delay * (self.retry_config.backoff_factor ** self.retry_count),
                self.retry_config.max_delay
            )
        
        # 添加随机抖动
        if self.retry_config.jitter:
            import random
            jitter = delay * 0.1 * random.random()  # 10%的抖动
            delay += jitter
        
        return utc_now() + timedelta(seconds=delay)
    
    def record_attempt(self, success: bool, error: Optional[str] = None):
        """记录发送尝试"""
        now = utc_now()
        self.last_attempt = now
        self.delivery_attempts.append(now)
        
        if success:
            self.status = MessageStatus.SENT
        else:
            if error:
                self.error_history.append(error)
            
            # 只有在这不是第一次尝试时才增加重试次数
            if self.status != MessageStatus.PENDING:
                self.retry_count += 1
            
            if self.can_retry():
                self.status = MessageStatus.RETRYING
                self.next_retry = self.calculate_next_retry_time()
            else:
                self.status = MessageStatus.DEAD_LETTER
    
    def mark_acknowledged(self):
        """标记为已确认"""
        self.status = MessageStatus.ACKNOWLEDGED
    
    def is_expired(self, ttl_seconds: Optional[float] = None) -> bool:
        """检查消息是否过期"""
        if not ttl_seconds:
            ttl_seconds = self.message.header.ttl or 300  # 默认5分钟
        
        elapsed = (utc_now() - self.created_at).total_seconds()
        return elapsed > ttl_seconds

@dataclass
class DeadLetterQueueConfig:
    """死信队列配置"""
    max_size: int = 10000
    retention_hours: int = 24
    auto_cleanup: bool = True
    cleanup_interval_minutes: int = 60

class ReliabilityManager:
    """消息可靠性管理器"""
    
    def __init__(
        self,
        default_retry_config: Optional[RetryConfig] = None,
        dlq_config: Optional[DeadLetterQueueConfig] = None,
        ack_timeout: float = 30.0,
        enable_persistence: bool = True
    ):
        self.default_retry_config = default_retry_config or RetryConfig()
        self.dlq_config = dlq_config or DeadLetterQueueConfig()
        self.ack_timeout = ack_timeout
        self.enable_persistence = enable_persistence
        
        # 消息存储
        self.pending_messages: Dict[str, ReliableMessage] = {}
        self.acknowledged_messages: Dict[str, ReliableMessage] = {}
        self.dead_letter_queue: Dict[str, ReliableMessage] = {}
        
        # 等待确认的消息
        self.awaiting_ack: Dict[str, ReliableMessage] = {}
        
        # 发送回调
        self.send_callback: Optional[Callable[[Message], bool]] = None
        
        # 统计信息
        self.stats = {
            "messages_sent": 0,
            "messages_acknowledged": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "messages_dead_lettered": 0,
            "total_retries": 0,
            "avg_delivery_time": 0.0
        }
        
        # 后台任务
        self.retry_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.ack_timeout_task: Optional[asyncio.Task] = None
        
        # 运行控制
        self.is_running = False
        
        logger.info("可靠性管理器初始化完成")
    
    async def start(self):
        """启动可靠性管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动后台任务
        self.retry_task = asyncio.create_task(self._retry_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.ack_timeout_task = asyncio.create_task(self._ack_timeout_loop())
        
        logger.info("可靠性管理器已启动")
    
    async def stop(self):
        """停止可靠性管理器"""
        self.is_running = False
        
        # 取消后台任务
        for task in [self.retry_task, self.cleanup_task, self.ack_timeout_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    raise
        
        logger.info("可靠性管理器已停止")
    
    def set_send_callback(self, callback: Callable[[Message], bool]):
        """设置消息发送回调"""
        self.send_callback = callback
    
    async def send_reliable_message(
        self,
        message: Message,
        retry_config: Optional[RetryConfig] = None,
        require_ack: bool = True
    ) -> str:
        """发送可靠消息"""
        message_id = str(uuid.uuid4())
        retry_config = retry_config or self.default_retry_config
        
        # 创建可靠消息包装
        reliable_msg = ReliableMessage(
            message=message,
            message_id=message_id,
            retry_config=retry_config,
            require_ack=require_ack
        )
        
        # 存储消息
        self.pending_messages[message_id] = reliable_msg
        
        # 尝试发送
        success = await self._attempt_send(reliable_msg)
        
        if success:
            if require_ack:
                # 等待确认
                self.awaiting_ack[message_id] = reliable_msg
            else:
                # 不需要确认，直接标记为成功
                reliable_msg.status = MessageStatus.ACKNOWLEDGED
                self.acknowledged_messages[message_id] = reliable_msg
                if message_id in self.pending_messages:
                    del self.pending_messages[message_id]
            
            self.stats["messages_sent"] += 1
        else:
            logger.warning(f"消息初次发送失败，将进入重试队列: {message_id}")
        
        return message_id
    
    async def _attempt_send(self, reliable_msg: ReliableMessage) -> bool:
        """尝试发送消息"""
        try:
            if not self.send_callback:
                logger.error("未设置发送回调函数")
                return False
            
            # 调用发送回调
            success = await self.send_callback(reliable_msg.message)
            
            # 记录发送尝试
            reliable_msg.record_attempt(success, None if success else "发送回调返回False")
            
            return success
            
        except Exception as e:
            error_msg = f"发送消息异常: {e}"
            logger.error(error_msg)
            reliable_msg.record_attempt(False, error_msg)
            return False
    
    def acknowledge_message(self, message_id: str) -> bool:
        """确认消息"""
        if message_id in self.awaiting_ack:
            reliable_msg = self.awaiting_ack[message_id]
            reliable_msg.mark_acknowledged()
            
            # 移动到已确认队列
            self.acknowledged_messages[message_id] = reliable_msg
            del self.awaiting_ack[message_id]
            
            if message_id in self.pending_messages:
                del self.pending_messages[message_id]
            
            # 更新统计
            self.stats["messages_acknowledged"] += 1
            delivery_time = (utc_now() - reliable_msg.created_at).total_seconds()
            self._update_avg_delivery_time(delivery_time)
            
            logger.debug(f"消息已确认: {message_id}")
            return True
        
        logger.warning(f"未找到等待确认的消息: {message_id}")
        return False
    
    def nack_message(self, message_id: str, reason: str = ""):
        """否认消息（触发重试）"""
        if message_id in self.awaiting_ack:
            reliable_msg = self.awaiting_ack[message_id]
            del self.awaiting_ack[message_id]
            
            # 记录失败并计算重试
            reliable_msg.record_attempt(False, reason or "收到NACK")
            
            if reliable_msg.status == MessageStatus.DEAD_LETTER:
                self._move_to_dead_letter_queue(reliable_msg)
            else:
                # 重新加入pending队列等待重试
                self.pending_messages[message_id] = reliable_msg
            
            logger.debug(f"消息被否认，状态: {reliable_msg.status.name}, 原因: {reason}")
    
    def _move_to_dead_letter_queue(self, reliable_msg: ReliableMessage):
        """移动消息到死信队列"""
        # 检查死信队列大小限制
        if len(self.dead_letter_queue) >= self.dlq_config.max_size:
            # 移除最老的消息
            oldest_id = min(self.dead_letter_queue.keys(), 
                          key=lambda k: self.dead_letter_queue[k].created_at)
            del self.dead_letter_queue[oldest_id]
        
        reliable_msg.status = MessageStatus.DEAD_LETTER
        self.dead_letter_queue[reliable_msg.message_id] = reliable_msg
        
        # 从pending队列移除
        if reliable_msg.message_id in self.pending_messages:
            del self.pending_messages[reliable_msg.message_id]
        
        self.stats["messages_dead_lettered"] += 1
        logger.warning(f"消息进入死信队列: {reliable_msg.message_id}, 重试次数: {reliable_msg.retry_count}")
    
    async def _retry_loop(self):
        """重试循环"""
        while self.is_running:
            try:
                await self._process_retries()
                await asyncio.sleep(1.0)  # 每秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"重试循环异常: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_retries(self):
        """处理重试队列"""
        retry_messages = []
        
        for message_id, reliable_msg in list(self.pending_messages.items()):
            # 检查需要重试的消息：PENDING (首次尝试) 或 RETRYING (后续重试)
            if reliable_msg.should_retry_now() and reliable_msg.status in [MessageStatus.PENDING, MessageStatus.RETRYING]:
                retry_messages.append(reliable_msg)
        
        for reliable_msg in retry_messages:
            logger.debug(f"重试消息: {reliable_msg.message_id}, 第{reliable_msg.retry_count + 1}次")
            success = await self._attempt_send(reliable_msg)
            
            if success:
                if reliable_msg.require_ack:
                    # 需要确认，移到等待确认队列
                    self.awaiting_ack[reliable_msg.message_id] = reliable_msg
                else:
                    # 不需要确认，直接标记为成功
                    reliable_msg.status = MessageStatus.ACKNOWLEDGED
                    self.acknowledged_messages[reliable_msg.message_id] = reliable_msg
                
                # 从pending队列移除
                if reliable_msg.message_id in self.pending_messages:
                    del self.pending_messages[reliable_msg.message_id]
                
                self.stats["messages_retried"] += 1
                self.stats["total_retries"] += reliable_msg.retry_count
            else:
                if reliable_msg.status == MessageStatus.DEAD_LETTER:
                    self._move_to_dead_letter_queue(reliable_msg)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._cleanup_expired_messages()
                # 每小时清理一次
                await asyncio.sleep(self.dlq_config.cleanup_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                await asyncio.sleep(300)  # 异常后等待5分钟
    
    async def _cleanup_expired_messages(self):
        """清理过期消息"""
        if not self.dlq_config.auto_cleanup:
            return
        
        now = utc_now()
        retention_cutoff = now - timedelta(hours=self.dlq_config.retention_hours)
        
        # 清理死信队列中的过期消息
        expired_dlq = []
        for message_id, reliable_msg in self.dead_letter_queue.items():
            if reliable_msg.created_at < retention_cutoff:
                expired_dlq.append(message_id)
        
        for message_id in expired_dlq:
            del self.dead_letter_queue[message_id]
        
        # 清理已确认消息中的过期消息
        expired_ack = []
        for message_id, reliable_msg in self.acknowledged_messages.items():
            if reliable_msg.created_at < retention_cutoff:
                expired_ack.append(message_id)
        
        for message_id in expired_ack:
            del self.acknowledged_messages[message_id]
        
        if expired_dlq or expired_ack:
            logger.info(f"清理过期消息: 死信队列 {len(expired_dlq)} 条, 已确认队列 {len(expired_ack)} 条")
    
    async def _ack_timeout_loop(self):
        """确认超时循环"""
        while self.is_running:
            try:
                await self._check_ack_timeouts()
                await asyncio.sleep(10.0)  # 每10秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"确认超时循环异常: {e}")
                await asyncio.sleep(30.0)
    
    async def _check_ack_timeouts(self):
        """检查确认超时"""
        timeout_messages = []
        now = utc_now()
        
        for message_id, reliable_msg in self.awaiting_ack.items():
            if reliable_msg.last_attempt:
                elapsed = (now - reliable_msg.last_attempt).total_seconds()
                if elapsed > self.ack_timeout:
                    timeout_messages.append(message_id)
        
        for message_id in timeout_messages:
            logger.warning(f"消息确认超时: {message_id}")
            self.nack_message(message_id, "确认超时")
    
    def _update_avg_delivery_time(self, delivery_time: float):
        """更新平均交付时间"""
        current_avg = self.stats["avg_delivery_time"]
        message_count = self.stats["messages_acknowledged"]
        
        if message_count <= 1:
            self.stats["avg_delivery_time"] = delivery_time
        else:
            self.stats["avg_delivery_time"] = (
                (current_avg * (message_count - 1) + delivery_time) / message_count
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "pending_messages": len(self.pending_messages),
            "awaiting_ack": len(self.awaiting_ack),
            "acknowledged_messages": len(self.acknowledged_messages),
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "is_running": self.is_running,
            **self.stats
        }
    
    def get_dead_letter_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取死信队列消息"""
        messages = []
        items = list(self.dead_letter_queue.items())
        
        if limit:
            items = items[:limit]
        
        for message_id, reliable_msg in items:
            messages.append({
                "message_id": message_id,
                "original_message": {
                    "sender_id": reliable_msg.message.sender_id,
                    "receiver_id": reliable_msg.message.receiver_id,
                    "message_type": reliable_msg.message.message_type.value,
                    "payload": reliable_msg.message.payload
                },
                "retry_count": reliable_msg.retry_count,
                "created_at": reliable_msg.created_at.isoformat(),
                "error_history": reliable_msg.error_history,
                "delivery_attempts": [dt.isoformat() for dt in reliable_msg.delivery_attempts]
            })
        
        return messages
    
    def replay_dead_letter_message(self, message_id: str) -> bool:
        """重放死信队列中的消息"""
        if message_id not in self.dead_letter_queue:
            logger.warning(f"死信队列中未找到消息: {message_id}")
            return False
        
        reliable_msg = self.dead_letter_queue[message_id]
        
        # 重置状态和重试计数
        reliable_msg.status = MessageStatus.PENDING
        reliable_msg.retry_count = 0
        reliable_msg.error_history.clear()
        reliable_msg.delivery_attempts.clear()
        reliable_msg.next_retry = None
        
        # 移回pending队列
        self.pending_messages[message_id] = reliable_msg
        del self.dead_letter_queue[message_id]
        
        logger.info(f"重放死信消息: {message_id}")
        return True
    
    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """获取消息状态"""
        # 查找消息
        reliable_msg = None
        location = "unknown"
        
        if message_id in self.pending_messages:
            reliable_msg = self.pending_messages[message_id]
            location = "pending"
        elif message_id in self.awaiting_ack:
            reliable_msg = self.awaiting_ack[message_id]
            location = "awaiting_ack"
        elif message_id in self.acknowledged_messages:
            reliable_msg = self.acknowledged_messages[message_id]
            location = "acknowledged"
        elif message_id in self.dead_letter_queue:
            reliable_msg = self.dead_letter_queue[message_id]
            location = "dead_letter_queue"
        
        if not reliable_msg:
            return None
        
        return {
            "message_id": message_id,
            "status": reliable_msg.status.name,
            "location": location,
            "retry_count": reliable_msg.retry_count,
            "max_retries": reliable_msg.max_retries,
            "created_at": reliable_msg.created_at.isoformat(),
            "last_attempt": reliable_msg.last_attempt.isoformat() if reliable_msg.last_attempt else None,
            "next_retry": reliable_msg.next_retry.isoformat() if reliable_msg.next_retry else None,
            "delivery_attempts": len(reliable_msg.delivery_attempts),
            "error_count": len(reliable_msg.error_history),
            "latest_error": reliable_msg.error_history[-1] if reliable_msg.error_history else None
        }
