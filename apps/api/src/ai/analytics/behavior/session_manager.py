"""
会话管理器

负责用户会话的生命周期管理、分割、聚合和重放功能。
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from collections import defaultdict
import json
from ..models import BehaviorEvent, UserSession, EventQueryFilter
from ..storage.event_store import EventStore

from src.core.logging import get_logger
logger = get_logger(__name__)

class SessionManager:
    """会话管理器"""
    
    def __init__(
        self,
        event_store: EventStore,
        session_timeout_minutes: int = 30,
        max_session_duration_hours: int = 8,
        session_gap_minutes: int = 15
    ):
        self.event_store = event_store
        self.session_timeout_minutes = session_timeout_minutes
        self.max_session_duration_hours = max_session_duration_hours
        self.session_gap_minutes = session_gap_minutes
        
        # 活跃会话缓存
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_last_activity: Dict[str, datetime] = {}
        
        # 会话统计
        self.stats = {
            'total_sessions': 0,
            'active_sessions_count': 0,
            'completed_sessions': 0,
            'timed_out_sessions': 0,
            'avg_session_duration_minutes': 0.0,
            'avg_events_per_session': 0.0
        }
        
        # 启动会话清理任务
        self._cleanup_task = None
        self._running = False
    
    async def start(self):
        """启动会话管理器"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = create_task_with_logging(self._periodic_cleanup())
        logger.info("会话管理器已启动")
    
    async def stop(self):
        """停止会话管理器"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                raise
        
        # 关闭所有活跃会话
        await self._close_all_active_sessions()
        logger.info("会话管理器已停止")
    
    async def process_event(self, event: BehaviorEvent) -> Optional[UserSession]:
        """处理事件并更新会话状态"""
        try:
            user_id = event.user_id
            session_id = event.session_id
            
            # 更新用户最后活动时间
            self.user_last_activity[user_id] = event.timestamp
            
            # 获取或创建会话
            session = await self._get_or_create_session(user_id, session_id, event.timestamp)
            
            # 更新会话信息
            session.events_count += 1
            session.end_time = event.timestamp
            
            # 检查是否需要分割会话
            if await self._should_split_session(session, event):
                # 关闭当前会话
                await self._close_session(session)
                
                # 创建新会话
                new_session_id = f"{user_id}_{int(event.timestamp.timestamp())}"
                session = await self._create_session(user_id, new_session_id, event.timestamp)
                session.events_count = 1
            
            # 计算交互质量分数
            await self._update_interaction_quality(session, event)
            
            return session
            
        except Exception as e:
            logger.error(f"处理事件时发生错误: {e}")
            return None
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """获取会话信息"""
        # 先从活跃会话缓存查找
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # 从存储查找已关闭的会话
        return await self._load_session_from_storage(session_id)
    
    async def get_user_sessions(
        self, 
        user_id: str, 
        limit: int = 50,
        include_active: bool = True
    ) -> List[UserSession]:
        """获取用户的所有会话"""
        sessions = []
        
        # 添加活跃会话
        if include_active:
            for session in self.active_sessions.values():
                if session.user_id == user_id:
                    sessions.append(session)
        
        # 从存储获取历史会话
        stored_sessions = await self.event_store.get_user_sessions(user_id, limit)
        
        # 转换为UserSession对象
        for session_data in stored_sessions:
            session = UserSession(
                session_id=session_data['session_id'],
                user_id=user_id,
                start_time=datetime.fromisoformat(session_data['start_time']),
                end_time=datetime.fromisoformat(session_data['end_time']),
                events_count=session_data['events_count']
            )
            sessions.append(session)
        
        # 按开始时间降序排序
        sessions.sort(key=lambda s: s.start_time, reverse=True)
        
        return sessions[:limit]
    
    async def get_session_replay_data(self, session_id: str) -> Dict[str, Any]:
        """获取会话重放数据"""
        try:
            # 获取会话信息
            session = await self.get_session(session_id)
            if not session:
                return {'error': '会话不存在'}
            
            # 查询会话的所有事件
            filter_params = EventQueryFilter(
                session_id=session_id,
                limit=10000  # 假设单个会话不会超过10k事件
            )
            
            events, total = await self.event_store.query_events(filter_params)
            
            # 构建重放数据
            replay_data = {
                'session': session.model_dump(mode="json"),
                'total_events': total,
                'events': events,
                'timeline': await self._build_session_timeline(events),
                'statistics': await self._calculate_session_statistics(events),
                'interaction_flow': await self._build_interaction_flow(events)
            }
            
            return replay_data
            
        except Exception as e:
            logger.error(f"获取会话重放数据失败: {e}")
            return {'error': str(e)}
    
    async def _get_or_create_session(
        self, 
        user_id: str, 
        session_id: str, 
        timestamp: datetime
    ) -> UserSession:
        """获取或创建会话"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        return await self._create_session(user_id, session_id, timestamp)
    
    async def _create_session(
        self, 
        user_id: str, 
        session_id: str, 
        timestamp: datetime
    ) -> UserSession:
        """创建新会话"""
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=timestamp,
            end_time=timestamp
        )
        
        self.active_sessions[session_id] = session
        self.stats['total_sessions'] += 1
        self.stats['active_sessions_count'] = len(self.active_sessions)
        
        logger.debug(f"创建新会话: {session_id} for user {user_id}")
        return session
    
    async def _should_split_session(self, session: UserSession, event: BehaviorEvent) -> bool:
        """判断是否需要分割会话"""
        # 检查会话间隔时间
        if session.end_time:
            time_gap = (event.timestamp - session.end_time).total_seconds() / 60
            if time_gap > self.session_gap_minutes:
                return True
        
        # 检查会话总时长
        session_duration = (event.timestamp - session.start_time).total_seconds() / 3600
        if session_duration > self.max_session_duration_hours:
            return True
        
        # 检查特定事件类型(如登出事件)
        if event.event_name in ['user_logout', 'session_end']:
            return True
        
        return False
    
    async def _close_session(self, session: UserSession):
        """关闭会话"""
        try:
            # 从活跃会话中移除
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            # 计算会话持续时间
            if session.end_time and session.start_time:
                duration_minutes = (session.end_time - session.start_time).total_seconds() / 60
                
                # 更新统计信息
                self._update_session_stats(duration_minutes, session.events_count)
            
            # 可选：将会话信息持久化到存储
            # await self._persist_session(session)
            
            self.stats['completed_sessions'] += 1
            self.stats['active_sessions_count'] = len(self.active_sessions)
            
            logger.debug(f"关闭会话: {session.session_id}")
            
        except Exception as e:
            logger.error(f"关闭会话失败: {e}")
    
    async def _close_all_active_sessions(self):
        """关闭所有活跃会话"""
        active_session_ids = list(self.active_sessions.keys())
        for session_id in active_session_ids:
            session = self.active_sessions[session_id]
            await self._close_session(session)
    
    async def _periodic_cleanup(self):
        """定期清理超时会话"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                await self._cleanup_timeout_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期清理异常: {e}")
    
    async def _cleanup_timeout_sessions(self):
        """清理超时会话"""
        now = utc_now()
        timeout_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # 检查会话是否超时
            if session.end_time:
                inactive_minutes = (now - session.end_time).total_seconds() / 60
                if inactive_minutes > self.session_timeout_minutes:
                    timeout_sessions.append(session)
        
        # 关闭超时会话
        for session in timeout_sessions:
            await self._close_session(session)
            self.stats['timed_out_sessions'] += 1
        
        if timeout_sessions:
            logger.info(f"清理了{len(timeout_sessions)}个超时会话")
    
    async def _update_interaction_quality(self, session: UserSession, event: BehaviorEvent):
        """更新交互质量分数"""
        try:
            # 基础质量分数计算逻辑
            quality_factors = []
            
            # 事件类型质量权重
            event_quality_weights = {
                'user_action': 1.0,
                'feedback_event': 0.9,
                'agent_response': 0.8,
                'system_event': 0.5,
                'error_event': 0.1
            }
            
            quality_factors.append(event_quality_weights.get(event.event_type.value, 0.5))
            
            # 响应时间质量
            if event.duration_ms is not None:
                if event.duration_ms < 1000:  # 小于1秒
                    quality_factors.append(1.0)
                elif event.duration_ms < 5000:  # 小于5秒
                    quality_factors.append(0.8)
                else:  # 超过5秒
                    quality_factors.append(0.5)
            
            # 计算平均质量分数
            if quality_factors:
                event_quality = sum(quality_factors) / len(quality_factors)
                
                # 更新会话总体质量分数(移动平均)
                if session.interaction_quality_score is None:
                    session.interaction_quality_score = event_quality
                else:
                    # 简单移动平均
                    alpha = 0.1  # 学习率
                    session.interaction_quality_score = (
                        (1 - alpha) * session.interaction_quality_score + 
                        alpha * event_quality
                    )
                
        except Exception as e:
            logger.error(f"更新交互质量分数失败: {e}")
    
    def _update_session_stats(self, duration_minutes: float, events_count: int):
        """更新会话统计信息"""
        completed = self.stats['completed_sessions']
        
        # 更新平均会话时长
        current_avg_duration = self.stats['avg_session_duration_minutes']
        self.stats['avg_session_duration_minutes'] = (
            (current_avg_duration * completed + duration_minutes) / (completed + 1)
        )
        
        # 更新平均每会话事件数
        current_avg_events = self.stats['avg_events_per_session']
        self.stats['avg_events_per_session'] = (
            (current_avg_events * completed + events_count) / (completed + 1)
        )
    
    async def _load_session_from_storage(self, session_id: str) -> Optional[UserSession]:
        """从存储加载会话信息"""
        # 这里需要实现从存储加载会话的逻辑
        # 由于我们主要存储事件，需要通过聚合事件来重建会话信息
        try:
            filter_params = EventQueryFilter(session_id=session_id, limit=1)
            events, total = await self.event_store.query_events(filter_params)
            
            if not events:
                return None
            
            # 获取会话的第一个和最后一个事件来构建会话信息
            first_event = events[0]
            
            # 查询最后一个事件
            filter_params.limit = 1
            filter_params.offset = total - 1 if total > 0 else 0
            last_events, _ = await self.event_store.query_events(filter_params)
            
            last_event = last_events[0] if last_events else first_event
            
            session = UserSession(
                session_id=session_id,
                user_id=first_event['user_id'],
                start_time=datetime.fromisoformat(first_event['timestamp']),
                end_time=datetime.fromisoformat(last_event['timestamp']),
                events_count=total
            )
            
            return session
            
        except Exception as e:
            logger.error(f"从存储加载会话失败: {e}")
            return None
    
    async def _build_session_timeline(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建会话时间线"""
        timeline = []
        
        for event in events:
            timeline_item = {
                'timestamp': event['timestamp'],
                'event_type': event['event_type'],
                'event_name': event['event_name'],
                'duration_ms': event.get('duration_ms'),
                'description': self._generate_event_description(event)
            }
            timeline.append(timeline_item)
        
        return timeline
    
    async def _calculate_session_statistics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算会话统计信息"""
        if not events:
            return {}
        
        # 事件类型统计
        event_type_counts = defaultdict(int)
        event_name_counts = defaultdict(int)
        durations = []
        
        for event in events:
            event_type_counts[event['event_type']] += 1
            event_name_counts[event['event_name']] += 1
            
            if event.get('duration_ms'):
                durations.append(event['duration_ms'])
        
        stats = {
            'total_events': len(events),
            'event_type_distribution': dict(event_type_counts),
            'top_event_names': dict(sorted(event_name_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'session_duration_seconds': 0,
            'avg_event_duration_ms': 0,
            'unique_event_types': len(event_type_counts),
            'unique_event_names': len(event_name_counts)
        }
        
        # 计算会话总时长
        if events:
            start_time = datetime.fromisoformat(events[0]['timestamp'])
            end_time = datetime.fromisoformat(events[-1]['timestamp'])
            stats['session_duration_seconds'] = (end_time - start_time).total_seconds()
        
        # 计算平均事件持续时间
        if durations:
            stats['avg_event_duration_ms'] = sum(durations) / len(durations)
        
        return stats
    
    async def _build_interaction_flow(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建交互流程"""
        flow = []
        
        # 按时间排序事件
        sorted_events = sorted(events, key=lambda e: e['timestamp'])
        
        for i, event in enumerate(sorted_events):
            flow_item = {
                'step': i + 1,
                'timestamp': event['timestamp'],
                'event_type': event['event_type'],
                'event_name': event['event_name'],
                'description': self._generate_event_description(event)
            }
            
            # 添加与上一步的时间间隔
            if i > 0:
                prev_time = datetime.fromisoformat(sorted_events[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(event['timestamp'])
                flow_item['time_since_previous_seconds'] = (curr_time - prev_time).total_seconds()
            
            flow.append(flow_item)
        
        return flow
    
    def _generate_event_description(self, event: Dict[str, Any]) -> str:
        """生成事件描述"""
        event_type = event['event_type']
        event_name = event['event_name']
        
        descriptions = {
            'user_action': f"用户执行了{event_name}操作",
            'agent_response': f"智能体响应: {event_name}",
            'system_event': f"系统事件: {event_name}",
            'error_event': f"发生错误: {event_name}",
            'feedback_event': f"用户反馈: {event_name}"
        }
        
        return descriptions.get(event_type, f"{event_type}: {event_name}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        return {
            **self.stats,
            'active_sessions_details': [
                {
                    'session_id': session.session_id,
                    'user_id': session.user_id,
                    'start_time': session.start_time.isoformat(),
                    'events_count': session.events_count,
                    'duration_minutes': (session.end_time - session.start_time).total_seconds() / 60 if session.end_time else 0
                }
                for session in list(self.active_sessions.values())
            ]
        }
