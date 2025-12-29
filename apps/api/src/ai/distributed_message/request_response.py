"""
请求-响应机制实现
支持异步请求-响应模式，消息关联和超时处理
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from .models import Message, MessageType, MessagePriority

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class PendingRequest:
    """待处理请求"""
    correlation_id: str
    sender_id: str
    message_type: MessageType
    created_at: datetime
    timeout: float
    future: asyncio.Future
    retry_count: int = 0
    max_retries: int = 3
    
    def is_expired(self) -> bool:
        """检查是否已超时"""
        elapsed = (utc_now() - self.created_at).total_seconds()
        return elapsed > self.timeout
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return self.retry_count < self.max_retries and not self.future.done()

class RequestResponseManager:
    """请求-响应管理器"""
    
    def __init__(self, default_timeout: float = 30.0, max_concurrent_requests: int = 1000):
        self.default_timeout = default_timeout
        self.max_concurrent_requests = max_concurrent_requests
        
        # 待处理请求
        self.pending_requests: Dict[str, PendingRequest] = {}
        
        # 回调处理器
        self.response_callbacks: Dict[MessageType, Callable[[Message], Any]] = {}
        
        # 请求处理器
        self.request_handlers: Dict[MessageType, Callable[[Message], Any]] = {}
        
        # 统计信息
        self.stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "requests_timed_out": 0,
            "requests_failed": 0,
            "requests_handled": 0,
            "avg_response_time": 0.0
        }
        
        # 后台清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 60.0  # 60秒清理一次
        
        # 线程池用于同步处理器
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        logger.info("请求-响应管理器初始化完成")
    
    def start_background_tasks(self):
        """启动后台任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
    
    async def stop_background_tasks(self):
        """停止后台任务"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                raise
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
    
    async def send_request(
        self,
        sender_function: Callable,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        max_retries: int = 3
    ) -> Optional[Message]:
        """
        发送请求并等待响应
        
        Args:
            sender_function: 实际发送消息的函数
            receiver_id: 接收者ID
            message_type: 消息类型
            payload: 消息负载
            timeout: 超时时间（秒）
            priority: 消息优先级
            max_retries: 最大重试次数
        
        Returns:
            响应消息或None（超时/失败）
        """
        
        if len(self.pending_requests) >= self.max_concurrent_requests:
            logger.warning("并发请求数达到上限，拒绝新请求")
            return None
        
        try:
            correlation_id = str(uuid.uuid4())
            request_timeout = timeout or self.default_timeout
            
            # 创建Future等待响应
            response_future = asyncio.Future()
            
            # 记录待处理请求
            pending_request = PendingRequest(
                correlation_id=correlation_id,
                sender_id=receiver_id,
                message_type=message_type,
                created_at=utc_now(),
                timeout=request_timeout,
                future=response_future,
                max_retries=max_retries
            )
            
            self.pending_requests[correlation_id] = pending_request
            
            try:
                # 发送请求消息
                success = await sender_function(
                    receiver_id=receiver_id,
                    message_type=message_type,
                    payload=payload,
                    correlation_id=correlation_id,
                    priority=priority
                )
                
                if not success:
                    logger.error("发送请求失败")
                    return None
                
                self.stats["requests_sent"] += 1
                
                # 等待响应
                start_time = time.time()
                response = await asyncio.wait_for(response_future, timeout=request_timeout)
                
                # 更新统计信息
                response_time = time.time() - start_time
                self._update_response_time_stats(response_time)
                self.stats["responses_received"] += 1
                
                logger.debug(f"收到响应 {correlation_id}，耗时: {response_time:.3f}s")
                return response
                
            except asyncio.TimeoutError:
                self.stats["requests_timed_out"] += 1
                logger.warning(f"请求超时: {correlation_id}")
                return None
            
            except Exception as e:
                self.stats["requests_failed"] += 1
                logger.error(f"请求失败: {e}")
                return None
                
        finally:
            # 清理pending request
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
    
    def handle_response(self, message: Message) -> bool:
        """处理收到的响应消息"""
        try:
            correlation_id = message.header.correlation_id
            
            if not correlation_id:
                logger.warning("响应消息缺少correlation_id")
                return False
            
            if correlation_id not in self.pending_requests:
                logger.debug(f"未找到对应的请求: {correlation_id}")
                return False
            
            pending_request = self.pending_requests[correlation_id]
            
            # 设置响应结果
            if not pending_request.future.done():
                pending_request.future.set_result(message)
                logger.debug(f"设置响应结果: {correlation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"处理响应消息失败: {e}")
            return False
    
    def register_request_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Union[Dict[str, Any], asyncio.Future]],
        is_async: bool = True
    ):
        """
        注册请求处理器
        
        Args:
            message_type: 消息类型
            handler: 处理函数，应返回响应数据或Future
            is_async: 是否为异步处理器
        """
        self.request_handlers[message_type] = {
            'handler': handler,
            'is_async': is_async
        }
        logger.info(f"注册请求处理器: {message_type.value}")
    
    async def handle_request(
        self,
        message: Message,
        reply_function: Callable
    ) -> bool:
        """
        处理收到的请求消息
        
        Args:
            message: 请求消息
            reply_function: 发送回复的函数
        
        Returns:
            是否成功处理
        """
        try:
            message_type = message.message_type
            
            if message_type not in self.request_handlers:
                # 发送未支持消息类型的错误响应
                error_payload = {
                    "error": "UNSUPPORTED_MESSAGE_TYPE",
                    "message": f"不支持的消息类型: {message_type.value}",
                    "supported_types": list(self.request_handlers.keys())
                }
                
                await reply_function(message, error_payload, MessageType.NACK)
                return False
            
            handler_info = self.request_handlers[message_type]
            handler = handler_info['handler']
            is_async = handler_info['is_async']
            
            start_time = time.time()
            
            try:
                # 执行处理器
                if is_async:
                    result = await handler(message)
                else:
                    # 在线程池中执行同步处理器
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(self.thread_pool, handler, message)
                
                # 发送成功响应
                if isinstance(result, dict):
                    response_payload = result
                else:
                    response_payload = {"result": result}
                
                await reply_function(message, response_payload, MessageType.ACK)
                
                # 更新统计信息
                processing_time = time.time() - start_time
                self.stats["requests_handled"] += 1
                
                logger.debug(f"处理请求完成: {message.header.message_id}, 耗时: {processing_time:.3f}s")
                return True
                
            except Exception as e:
                # 发送错误响应
                error_payload = {
                    "error": "PROCESSING_ERROR",
                    "message": str(e),
                    "request_id": message.header.message_id
                }
                
                await reply_function(message, error_payload, MessageType.NACK)
                logger.error(f"处理请求失败: {e}")
                return False
                
        except Exception as e:
            logger.error(f"请求处理器执行失败: {e}")
            return False
    
    def register_response_callback(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Any]
    ):
        """注册响应回调处理器"""
        self.response_callbacks[message_type] = callback
        logger.info(f"注册响应回调: {message_type.value}")
    
    async def handle_callback_response(self, message: Message) -> bool:
        """处理回调响应"""
        try:
            message_type = message.message_type
            
            if message_type in self.response_callbacks:
                callback = self.response_callbacks[message_type]
                
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"处理回调响应失败: {e}")
            return False
    
    async def _cleanup_expired_requests(self):
        """清理过期的请求"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = utc_now()
                expired_requests = []
                
                for correlation_id, request in self.pending_requests.items():
                    if request.is_expired():
                        if not request.future.done():
                            request.future.cancel()
                        expired_requests.append(correlation_id)
                
                # 移除过期请求
                for correlation_id in expired_requests:
                    if correlation_id in self.pending_requests:
                        del self.pending_requests[correlation_id]
                        self.stats["requests_timed_out"] += 1
                
                if expired_requests:
                    logger.debug(f"清理过期请求: {len(expired_requests)}个")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理过期请求失败: {e}")
    
    def _update_response_time_stats(self, response_time: float):
        """更新响应时间统计"""
        current_avg = self.stats["avg_response_time"]
        response_count = self.stats["responses_received"]
        
        if response_count <= 1:
            self.stats["avg_response_time"] = response_time
        else:
            # 计算新的平均响应时间
            self.stats["avg_response_time"] = (
                (current_avg * (response_count - 1) + response_time) / response_count
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取请求-响应统计信息"""
        return {
            "pending_requests": len(self.pending_requests),
            "request_handlers": len(self.request_handlers),
            "response_callbacks": len(self.response_callbacks),
            **self.stats
        }
    
    def get_pending_requests_info(self) -> List[Dict[str, Any]]:
        """获取待处理请求信息"""
        requests_info = []
        
        for correlation_id, request in self.pending_requests.items():
            elapsed = (utc_now() - request.created_at).total_seconds()
            requests_info.append({
                "correlation_id": correlation_id,
                "sender_id": request.sender_id,
                "message_type": request.message_type.value,
                "created_at": request.created_at.isoformat(),
                "elapsed": elapsed,
                "timeout": request.timeout,
                "is_expired": request.is_expired(),
                "retry_count": request.retry_count,
                "max_retries": request.max_retries
            })
        
        return requests_info
