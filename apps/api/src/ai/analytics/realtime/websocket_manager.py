"""
WebSocket实时数据推送管理器

提供实时用户行为分析数据的WebSocket推送功能。
"""

import asyncio
import json
from typing import Dict, Set, List, Optional, Any, Callable
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
import logging
from contextlib import asynccontextmanager

from ..models import BehaviorEvent, AnomalyDetection
from ..behavior.anomaly_detection import AnomalyDetectionEngine
from ..storage.event_store import EventStore

logger = logging.getLogger(__name__)

@dataclass
class ConnectionInfo:
    """WebSocket连接信息"""
    websocket: WebSocket
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subscriptions: Set[str] = None
    connected_at: datetime = None
    last_ping: datetime = None

    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = set()
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()
        if self.last_ping is None:
            self.last_ping = datetime.utcnow()

@dataclass
class RealtimeMessage:
    """实时消息"""
    type: str  # 'event', 'anomaly', 'pattern', 'stats'
    data: Any
    timestamp: datetime = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # subscription_type -> connection_ids
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.broadcast_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0
        }
    
    async def start(self):
        """启动WebSocket管理器"""
        if self.running:
            return
        
        self.running = True
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("WebSocket管理器已启动")
    
    async def stop(self):
        """停止WebSocket管理器"""
        self.running = False
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("WebSocket管理器已停止")
    
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     user_id: Optional[str] = None, session_id: Optional[str] = None):
        """建立WebSocket连接"""
        await websocket.accept()
        
        connection_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id
        )
        
        self.active_connections[connection_id] = connection_info
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.active_connections)
        
        logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}")
        
        # 发送欢迎消息
        await self._send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            
            # 从所有订阅中移除
            for subscription_type, subscribers in self.subscriptions.items():
                subscribers.discard(connection_id)
            
            # 关闭连接
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.warning(f"关闭WebSocket连接失败: {e}")
            
            del self.active_connections[connection_id]
            self.stats["active_connections"] = len(self.active_connections)
            
            logger.info(f"WebSocket连接断开: {connection_id}")
    
    async def subscribe(self, connection_id: str, subscription_type: str):
        """订阅特定类型的消息"""
        if connection_id not in self.active_connections:
            raise ValueError(f"连接不存在: {connection_id}")
        
        connection = self.active_connections[connection_id]
        connection.subscriptions.add(subscription_type)
        
        if subscription_type not in self.subscriptions:
            self.subscriptions[subscription_type] = set()
        
        self.subscriptions[subscription_type].add(connection_id)
        
        logger.info(f"连接 {connection_id} 订阅: {subscription_type}")
        
        # 确认订阅消息
        await self._send_to_connection(connection_id, {
            "type": "subscription_confirmed",
            "subscription_type": subscription_type,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def unsubscribe(self, connection_id: str, subscription_type: str):
        """取消订阅"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.subscriptions.discard(subscription_type)
        
        if subscription_type in self.subscriptions:
            self.subscriptions[subscription_type].discard(connection_id)
        
        logger.info(f"连接 {connection_id} 取消订阅: {subscription_type}")
    
    async def broadcast_message(self, message: RealtimeMessage):
        """广播消息到订阅者"""
        await self.message_queue.put(message)
    
    async def send_to_user(self, user_id: str, message: RealtimeMessage):
        """发送消息给特定用户的所有连接"""
        message.user_id = user_id
        
        for connection_id, connection in self.active_connections.items():
            if connection.user_id == user_id:
                await self._send_to_connection(connection_id, message.to_dict())
    
    async def send_to_session(self, session_id: str, message: RealtimeMessage):
        """发送消息给特定会话的所有连接"""
        message.session_id = session_id
        
        for connection_id, connection in self.active_connections.items():
            if connection.session_id == session_id:
                await self._send_to_connection(connection_id, message.to_dict())
    
    async def _broadcast_loop(self):
        """消息广播循环"""
        while self.running:
            try:
                # 等待消息
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # 获取订阅该消息类型的连接
                subscribers = self.subscriptions.get(message.type, set())
                
                if not subscribers:
                    continue
                
                # 并发发送消息
                tasks = []
                for connection_id in subscribers:
                    if connection_id in self.active_connections:
                        task = self._send_to_connection(connection_id, message.to_dict())
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.TimeoutError:
                # 定期心跳检查
                await self._heartbeat_check()
                continue
            except Exception as e:
                logger.error(f"消息广播循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _send_to_connection(self, connection_id: str, data: Dict[str, Any]):
        """发送数据到特定连接"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(data, default=str))
            self.stats["messages_sent"] += 1
            
        except Exception as e:
            logger.error(f"发送消息失败 {connection_id}: {e}")
            self.stats["messages_failed"] += 1
            
            # 连接可能已断开，移除连接
            await self.disconnect(connection_id)
    
    async def _heartbeat_check(self):
        """心跳检查"""
        now = datetime.utcnow()
        timeout_connections = []
        
        for connection_id, connection in self.active_connections.items():
            # 检查是否超时（5分钟无心跳）
            if (now - connection.last_ping).total_seconds() > 300:
                timeout_connections.append(connection_id)
        
        # 移除超时连接
        for connection_id in timeout_connections:
            await self.disconnect(connection_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "subscription_stats": {
                sub_type: len(subscribers)
                for sub_type, subscribers in self.subscriptions.items()
            }
        }

class RealtimeAnalyticsManager:
    """实时分析管理器"""
    
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.anomaly_engine = AnomalyDetectionEngine()
        self.event_store = EventStore()
        self.running = False
        self.analysis_task: Optional[asyncio.Task] = None
        
        # 实时分析配置
        self.analysis_interval = 10  # 10秒分析间隔
        self.recent_events_window = 300  # 5分钟窗口
    
    async def start(self):
        """启动实时分析管理器"""
        await self.websocket_manager.start()
        self.running = True
        self.analysis_task = asyncio.create_task(self._realtime_analysis_loop())
        logger.info("实时分析管理器已启动")
    
    async def stop(self):
        """停止实时分析管理器"""
        self.running = False
        
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        await self.websocket_manager.stop()
        logger.info("实时分析管理器已停止")
    
    async def handle_new_event(self, event: BehaviorEvent):
        """处理新事件"""
        # 广播事件
        message = RealtimeMessage(
            type="event",
            data=asdict(event),
            user_id=event.user_id,
            session_id=event.session_id
        )
        await self.websocket_manager.broadcast_message(message)
        
        # 实时异常检测
        await self._check_realtime_anomaly(event)
    
    async def _realtime_analysis_loop(self):
        """实时分析循环"""
        while self.running:
            try:
                await self._perform_realtime_analysis()
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"实时分析循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _perform_realtime_analysis(self):
        """执行实时分析"""
        try:
            # 获取最近的事件
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=self.recent_events_window)
            
            recent_events = await self.event_store.get_events_in_timerange(
                start_time, end_time
            )
            
            if not recent_events:
                return
            
            # 生成实时统计
            stats = await self._generate_realtime_stats(recent_events)
            
            # 广播统计信息
            message = RealtimeMessage(
                type="stats",
                data=stats
            )
            await self.websocket_manager.broadcast_message(message)
            
        except Exception as e:
            logger.error(f"实时分析执行失败: {e}")
    
    async def _check_realtime_anomaly(self, event: BehaviorEvent):
        """实时异常检测"""
        try:
            # 获取用户最近的行为历史
            user_recent_events = await self.event_store.get_user_recent_events(
                event.user_id, 
                limit=100
            )
            
            # 执行快速异常检测
            anomalies = await self.anomaly_engine.detect_realtime_anomalies([event])
            
            if anomalies:
                # 广播异常信息
                for anomaly in anomalies:
                    message = RealtimeMessage(
                        type="anomaly",
                        data=asdict(anomaly),
                        user_id=event.user_id,
                        session_id=event.session_id
                    )
                    await self.websocket_manager.broadcast_message(message)
            
        except Exception as e:
            logger.error(f"实时异常检测失败: {e}")
    
    async def _generate_realtime_stats(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """生成实时统计信息"""
        if not events:
            return {}
        
        # 基础统计
        total_events = len(events)
        unique_users = len(set(event.user_id for event in events))
        unique_sessions = len(set(event.session_id for event in events if event.session_id))
        
        # 事件类型分布
        event_type_counts = {}
        for event in events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        # 时间分布（按小时）
        hour_counts = {}
        for event in events:
            hour = event.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        return {
            "window_duration_seconds": self.recent_events_window,
            "total_events": total_events,
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "events_per_minute": total_events / (self.recent_events_window / 60),
            "event_type_distribution": event_type_counts,
            "hourly_distribution": hour_counts,
            "most_active_hour": max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
        }

# 全局实例
realtime_manager = RealtimeAnalyticsManager()

@asynccontextmanager
async def get_realtime_manager():
    """获取实时分析管理器"""
    yield realtime_manager