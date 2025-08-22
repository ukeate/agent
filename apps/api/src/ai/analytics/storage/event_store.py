"""
事件存储模块

负责行为事件的持久化存储、查询和管理。
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from core.database import get_db_session
from ..models import BehaviorEvent, BulkEventRequest, EventQueryFilter

logger = logging.getLogger(__name__)


class EventStore:
    """事件存储器"""
    
    def __init__(self):
        self.table_name = "behavior_events"
        self._ensure_table_created = False
    
    async def _ensure_table(self):
        """确保事件表存在"""
        if self._ensure_table_created:
            return
        
        try:
            async with get_db_session() as session:
                # 创建事件表
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    event_id VARCHAR(36) PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    event_name VARCHAR(255) NOT NULL,
                    event_data JSONB,
                    context JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    duration_ms INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    
                    -- 创建索引
                    INDEX idx_{self.table_name}_user_id (user_id),
                    INDEX idx_{self.table_name}_session_id (session_id),
                    INDEX idx_{self.table_name}_timestamp (timestamp),
                    INDEX idx_{self.table_name}_event_type (event_type),
                    INDEX idx_{self.table_name}_event_name (event_name),
                    INDEX idx_{self.table_name}_composite (user_id, timestamp DESC)
                );
                """
                
                await session.execute(text(create_table_sql))
                await session.commit()
                
                # 创建分区表(按月分区)
                await self._create_partitions(session)
                
                self._ensure_table_created = True
                logger.info(f"事件表 {self.table_name} 创建完成")
                
        except Exception as e:
            logger.error(f"创建事件表失败: {e}")
            raise
    
    async def _create_partitions(self, session: AsyncSession):
        """创建分区表"""
        try:
            # 创建当前月和下个月的分区
            current_date = datetime.utcnow()
            for month_offset in range(0, 3):  # 当前月、下月、下下月
                target_date = current_date.replace(day=1) + timedelta(days=32 * month_offset)
                target_date = target_date.replace(day=1)
                
                partition_name = f"{self.table_name}_{target_date.strftime('%Y_%m')}"
                start_date = target_date
                end_date = (target_date + timedelta(days=32)).replace(day=1)
                
                partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF {self.table_name}
                FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}');
                """
                
                await session.execute(text(partition_sql))
            
        except Exception as e:
            logger.warning(f"创建分区失败(可能已存在): {e}")
    
    async def store_event(self, event: BehaviorEvent) -> bool:
        """存储单个事件"""
        await self._ensure_table()
        
        try:
            async with get_db_session() as session:
                insert_sql = f"""
                INSERT INTO {self.table_name} (
                    event_id, session_id, user_id, event_type, event_name,
                    event_data, context, timestamp, duration_ms
                ) VALUES (
                    :event_id, :session_id, :user_id, :event_type, :event_name,
                    :event_data, :context, :timestamp, :duration_ms
                )
                """
                
                await session.execute(text(insert_sql), {
                    'event_id': event.event_id,
                    'session_id': event.session_id,
                    'user_id': event.user_id,
                    'event_type': event.event_type.value,
                    'event_name': event.event_name,
                    'event_data': json.dumps(event.event_data),
                    'context': json.dumps(event.context),
                    'timestamp': event.timestamp,
                    'duration_ms': event.duration_ms
                })
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"存储事件失败: {e}")
            return False
    
    async def store_events_batch(self, bulk_request: BulkEventRequest) -> bool:
        """批量存储事件"""
        await self._ensure_table()
        
        if not bulk_request.events:
            return True
        
        try:
            async with get_db_session() as session:
                # 准备批量插入数据
                events_data = []
                for event in bulk_request.events:
                    events_data.append({
                        'event_id': event.event_id,
                        'session_id': event.session_id,
                        'user_id': event.user_id,
                        'event_type': event.event_type.value,
                        'event_name': event.event_name,
                        'event_data': json.dumps(event.event_data),
                        'context': json.dumps(event.context),
                        'timestamp': event.timestamp,
                        'duration_ms': event.duration_ms
                    })
                
                # 执行批量插入
                insert_sql = f"""
                INSERT INTO {self.table_name} (
                    event_id, session_id, user_id, event_type, event_name,
                    event_data, context, timestamp, duration_ms
                ) VALUES (
                    :event_id, :session_id, :user_id, :event_type, :event_name,
                    :event_data, :context, :timestamp, :duration_ms
                )
                """
                
                await session.execute(text(insert_sql), events_data)
                await session.commit()
                
                logger.debug(f"批量存储{len(bulk_request.events)}个事件成功")
                return True
                
        except Exception as e:
            logger.error(f"批量存储事件失败: {e}")
            return False
    
    async def query_events(
        self, 
        filter_params: EventQueryFilter
    ) -> Tuple[List[Dict[str, Any]], int]:
        """查询事件"""
        await self._ensure_table()
        
        try:
            async with get_db_session() as session:
                # 构建WHERE条件
                where_conditions = []
                params = {}
                
                if filter_params.user_id:
                    where_conditions.append("user_id = :user_id")
                    params['user_id'] = filter_params.user_id
                
                if filter_params.session_id:
                    where_conditions.append("session_id = :session_id")
                    params['session_id'] = filter_params.session_id
                
                if filter_params.event_types:
                    event_types = [et.value for et in filter_params.event_types]
                    where_conditions.append("event_type = ANY(:event_types)")
                    params['event_types'] = event_types
                
                if filter_params.event_names:
                    where_conditions.append("event_name = ANY(:event_names)")
                    params['event_names'] = filter_params.event_names
                
                if filter_params.start_time:
                    where_conditions.append("timestamp >= :start_time")
                    params['start_time'] = filter_params.start_time
                
                if filter_params.end_time:
                    where_conditions.append("timestamp <= :end_time")
                    params['end_time'] = filter_params.end_time
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # 查询总数
                count_sql = f"""
                SELECT COUNT(*) as total
                FROM {self.table_name}
                WHERE {where_clause}
                """
                
                count_result = await session.execute(text(count_sql), params)
                total_count = count_result.scalar()
                
                # 查询数据
                query_sql = f"""
                SELECT 
                    event_id, session_id, user_id, event_type, event_name,
                    event_data, context, timestamp, duration_ms, created_at
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
                """
                
                params.update({
                    'limit': filter_params.limit,
                    'offset': filter_params.offset
                })
                
                result = await session.execute(text(query_sql), params)
                events = []
                
                for row in result:
                    event_dict = {
                        'event_id': row.event_id,
                        'session_id': row.session_id,
                        'user_id': row.user_id,
                        'event_type': row.event_type,
                        'event_name': row.event_name,
                        'event_data': json.loads(row.event_data or '{}'),
                        'context': json.loads(row.context or '{}'),
                        'timestamp': row.timestamp.isoformat(),
                        'duration_ms': row.duration_ms,
                        'created_at': row.created_at.isoformat() if row.created_at else None
                    }
                    events.append(event_dict)
                
                return events, total_count
                
        except Exception as e:
            logger.error(f"查询事件失败: {e}")
            return [], 0
    
    async def get_event_statistics(
        self, 
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取事件统计信息"""
        await self._ensure_table()
        
        try:
            async with get_db_session() as session:
                # 构建WHERE条件
                where_conditions = []
                params = {}
                
                if user_id:
                    where_conditions.append("user_id = :user_id")
                    params['user_id'] = user_id
                
                if start_time:
                    where_conditions.append("timestamp >= :start_time")
                    params['start_time'] = start_time
                
                if end_time:
                    where_conditions.append("timestamp <= :end_time")
                    params['end_time'] = end_time
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # 综合统计查询
                stats_sql = f"""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    COUNT(DISTINCT event_type) as unique_event_types,
                    COUNT(DISTINCT event_name) as unique_event_names,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(timestamp) as earliest_event,
                    MAX(timestamp) as latest_event
                FROM {self.table_name}
                WHERE {where_clause}
                """
                
                stats_result = await session.execute(text(stats_sql), params)
                stats_row = stats_result.first()
                
                # 事件类型统计
                type_stats_sql = f"""
                SELECT event_type, COUNT(*) as count
                FROM {self.table_name}
                WHERE {where_clause}
                GROUP BY event_type
                ORDER BY count DESC
                """
                
                type_stats_result = await session.execute(text(type_stats_sql), params)
                event_type_stats = {row.event_type: row.count for row in type_stats_result}
                
                # 每日事件统计
                daily_stats_sql = f"""
                SELECT 
                    DATE(timestamp) as event_date,
                    COUNT(*) as count
                FROM {self.table_name}
                WHERE {where_clause}
                GROUP BY DATE(timestamp)
                ORDER BY event_date DESC
                LIMIT 30
                """
                
                daily_stats_result = await session.execute(text(daily_stats_sql), params)
                daily_stats = [
                    {'date': row.event_date.isoformat(), 'count': row.count}
                    for row in daily_stats_result
                ]
                
                return {
                    'total_events': stats_row.total_events or 0,
                    'unique_users': stats_row.unique_users or 0,
                    'unique_sessions': stats_row.unique_sessions or 0,
                    'unique_event_types': stats_row.unique_event_types or 0,
                    'unique_event_names': stats_row.unique_event_names or 0,
                    'avg_duration_ms': float(stats_row.avg_duration_ms or 0),
                    'earliest_event': stats_row.earliest_event.isoformat() if stats_row.earliest_event else None,
                    'latest_event': stats_row.latest_event.isoformat() if stats_row.latest_event else None,
                    'event_type_distribution': event_type_stats,
                    'daily_stats': daily_stats
                }
                
        except Exception as e:
            logger.error(f"获取事件统计失败: {e}")
            return {}
    
    async def cleanup_old_events(self, retention_days: int = 30) -> int:
        """清理过期事件"""
        await self._ensure_table()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with get_db_session() as session:
                delete_sql = f"""
                DELETE FROM {self.table_name}
                WHERE timestamp < :cutoff_date
                """
                
                result = await session.execute(text(delete_sql), {'cutoff_date': cutoff_date})
                deleted_count = result.rowcount
                await session.commit()
                
                logger.info(f"清理了{deleted_count}个过期事件(早于{cutoff_date.isoformat()})")
                return deleted_count
                
        except Exception as e:
            logger.error(f"清理过期事件失败: {e}")
            return 0
    
    async def get_user_sessions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取用户会话列表"""
        await self._ensure_table()
        
        try:
            async with get_db_session() as session:
                sessions_sql = f"""
                SELECT 
                    session_id,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    COUNT(*) as events_count,
                    COUNT(DISTINCT event_name) as unique_events
                FROM {self.table_name}
                WHERE user_id = :user_id
                GROUP BY session_id
                ORDER BY start_time DESC
                LIMIT :limit
                """
                
                result = await session.execute(text(sessions_sql), {
                    'user_id': user_id,
                    'limit': limit
                })
                
                sessions = []
                for row in result:
                    duration_seconds = 0
                    if row.start_time and row.end_time:
                        duration_seconds = (row.end_time - row.start_time).total_seconds()
                    
                    session_data = {
                        'session_id': row.session_id,
                        'start_time': row.start_time.isoformat(),
                        'end_time': row.end_time.isoformat(),
                        'duration_seconds': duration_seconds,
                        'events_count': row.events_count,
                        'unique_events': row.unique_events
                    }
                    sessions.append(session_data)
                
                return sessions
                
        except Exception as e:
            logger.error(f"获取用户会话失败: {e}")
            return []