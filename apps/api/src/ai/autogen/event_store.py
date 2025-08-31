"""
事件持久化和重播系统
实现事件存储、检索和重播功能
"""
import asyncio
import json
import uuid
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import structlog
try:
    import asyncpg
except ImportError:
    asyncpg = None
try:
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis
    except ImportError:
        aioredis = None

from .events import Event, EventType, EventPriority
from .event_processors import AsyncEventProcessingEngine, EventContext

logger = structlog.get_logger(__name__)


class EventStore:
    """事件存储系统"""
    
    def __init__(self, redis_client=None, postgres_pool=None):
        self.redis = redis_client
        self.postgres = postgres_pool
        self.stream_prefix = "events:"
        self.event_ttl = 86400  # 默认TTL为1天
        self.max_stream_length = 10000  # Redis Stream最大长度
        
        # 事件版本管理
        self.event_version = "1.0"
        self.schema_versions = {
            "1.0": self._serialize_v1,
            "2.0": self._serialize_v2  # 预留未来版本
        }
        
        self.stats = {
            "events_stored": 0,
            "events_retrieved": 0,
            "storage_errors": 0
        }
    
    async def initialize(self) -> None:
        """初始化事件存储"""
        if self.postgres:
            await self._create_tables()
        
        logger.info("事件存储初始化完成")
    
    async def _create_tables(self) -> None:
        """创建事件存储表"""
        try:
            async with self.postgres.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id UUID PRIMARY KEY,
                        type VARCHAR(100) NOT NULL,
                        source VARCHAR(200),
                        target VARCHAR(200),
                        conversation_id UUID,
                        session_id UUID,
                        correlation_id UUID,
                        priority VARCHAR(20),
                        data JSONB,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        version VARCHAR(10) DEFAULT '1.0',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        INDEX idx_events_timestamp (timestamp),
                        INDEX idx_events_type (type),
                        INDEX idx_events_conversation (conversation_id),
                        INDEX idx_events_correlation (correlation_id)
                    )
                """)
                
                # 创建事件快照表（用于长期存储）
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_snapshots (
                        id UUID PRIMARY KEY,
                        aggregate_id VARCHAR(200) NOT NULL,
                        aggregate_type VARCHAR(100) NOT NULL,
                        snapshot_data JSONB NOT NULL,
                        event_sequence BIGINT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        INDEX idx_snapshots_aggregate (aggregate_id, aggregate_type),
                        INDEX idx_snapshots_sequence (event_sequence)
                    )
                """)
                
                # 创建死信队列表
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS dead_letter_events (
                        id UUID PRIMARY KEY,
                        original_event_id UUID,
                        event_data JSONB NOT NULL,
                        error_message TEXT,
                        retry_count INT DEFAULT 0,
                        last_retry_at TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                logger.info("事件存储表创建成功")
                
        except Exception as e:
            logger.error("创建事件存储表失败", error=str(e))
            raise
    
    async def append_event(self, event: Event) -> str:
        """追加事件到存储"""
        event_id = event.id if hasattr(event, 'id') else str(uuid.uuid4())
        
        try:
            # 序列化事件
            event_data = self._serialize_event(event)
            
            # 并行写入Redis和PostgreSQL
            tasks = []
            
            if self.redis:
                tasks.append(self._store_to_redis(event, event_data))
            
            if self.postgres:
                tasks.append(self._store_to_postgres(event_id, event, event_data))
            
            if tasks:
                await asyncio.gather(*tasks)
            
            self.stats["events_stored"] += 1
            
            logger.debug(
                "事件存储成功",
                event_id=event_id,
                event_type=event.type.value if hasattr(event.type, 'value') else event.type
            )
            
            return event_id
            
        except Exception as e:
            self.stats["storage_errors"] += 1
            logger.error("事件存储失败", event_id=event_id, error=str(e))
            raise
    
    async def _store_to_redis(self, event: Event, event_data: Dict[str, Any]) -> None:
        """存储事件到Redis"""
        try:
            # 使用Redis Streams存储实时事件
            stream_key = f"{self.stream_prefix}{event.type.value if hasattr(event.type, 'value') else event.type}"
            
            await self.redis.xadd(
                stream_key,
                event_data,
                maxlen=self.max_stream_length,
                approximate=True  # 使用近似修剪以提高性能
            )
            
            # 设置事件索引（用于快速查找）
            if hasattr(event, 'correlation_id') and event.correlation_id:
                await self.redis.setex(
                    f"event:correlation:{event.correlation_id}",
                    self.event_ttl,
                    json.dumps(event_data)
                )
            
        except Exception as e:
            logger.error("Redis存储失败", error=str(e))
            raise
    
    async def _store_to_postgres(self, event_id: str, event: Event, event_data: Dict[str, Any]) -> None:
        """存储事件到PostgreSQL"""
        try:
            async with self.postgres.acquire() as conn:
                await conn.execute("""
                    INSERT INTO events (
                        id, type, source, target, conversation_id, session_id,
                        correlation_id, priority, data, timestamp, version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    uuid.UUID(event_id),
                    event.type.value if hasattr(event.type, 'value') else str(event.type),
                    getattr(event, 'source', None),
                    getattr(event, 'target', None),
                    uuid.UUID(event.conversation_id) if hasattr(event, 'conversation_id') and event.conversation_id else None,
                    uuid.UUID(event.session_id) if hasattr(event, 'session_id') and event.session_id else None,
                    uuid.UUID(event.correlation_id) if hasattr(event, 'correlation_id') and event.correlation_id else None,
                    event.priority.value if hasattr(event, 'priority') and hasattr(event.priority, 'value') else 'normal',
                    json.dumps(event.data) if hasattr(event, 'data') else '{}',
                    event.timestamp if hasattr(event, 'timestamp') else utc_now(),
                    self.event_version
                )
                
        except Exception as e:
            logger.error("PostgreSQL存储失败", error=str(e))
            raise
    
    def _serialize_event(self, event: Event) -> Dict[str, Any]:
        """序列化事件"""
        serializer = self.schema_versions.get(self.event_version, self._serialize_v1)
        return serializer(event)
    
    def _serialize_v1(self, event: Event) -> Dict[str, Any]:
        """V1版本序列化"""
        return {
            "id": event.id if hasattr(event, 'id') else str(uuid.uuid4()),
            "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
            "source": getattr(event, 'source', None),
            "target": getattr(event, 'target', None),
            "conversation_id": getattr(event, 'conversation_id', None),
            "session_id": getattr(event, 'session_id', None),
            "correlation_id": getattr(event, 'correlation_id', None),
            "priority": event.priority.value if hasattr(event, 'priority') and hasattr(event.priority, 'value') else 'normal',
            "data": event.data if hasattr(event, 'data') else {},
            "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else utc_now().isoformat(),
            "version": self.event_version
        }
    
    def _serialize_v2(self, event: Event) -> Dict[str, Any]:
        """V2版本序列化（预留）"""
        # 未来版本的序列化逻辑
        return self._serialize_v1(event)
    
    async def get_event(self, event_id: str) -> Optional[Event]:
        """获取单个事件"""
        try:
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM events WHERE id = $1",
                        uuid.UUID(event_id)
                    )
                    
                    if row:
                        return self._deserialize_event(dict(row))
            
            return None
            
        except Exception as e:
            logger.error("获取事件失败", event_id=event_id, error=str(e))
            return None
    
    async def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """根据关联ID获取事件"""
        events = []
        
        try:
            # 先从Redis获取
            if self.redis:
                key = f"event:correlation:{correlation_id}"
                data = await self.redis.get(key)
                if data:
                    event_data = json.loads(data)
                    event = self._deserialize_event(event_data)
                    if event:
                        events.append(event)
            
            # 从PostgreSQL获取完整列表
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM events WHERE correlation_id = $1 ORDER BY timestamp",
                        uuid.UUID(correlation_id)
                    )
                    
                    for row in rows:
                        event = self._deserialize_event(dict(row))
                        if event:
                            events.append(event)
            
            self.stats["events_retrieved"] += len(events)
            return events
            
        except Exception as e:
            logger.error("获取关联事件失败", correlation_id=correlation_id, error=str(e))
            return []
    
    async def replay_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[EventType]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Event]:
        """重播指定时间范围的事件"""
        events = []
        
        try:
            if not self.postgres:
                logger.warning("PostgreSQL未配置，无法重播事件")
                return events
            
            async with self.postgres.acquire() as conn:
                # 构建查询
                query = """
                    SELECT * FROM events 
                    WHERE timestamp BETWEEN $1 AND $2
                """
                params = [start_time, end_time]
                
                # 添加事件类型过滤
                if event_types:
                    type_values = [t.value if hasattr(t, 'value') else str(t) for t in event_types]
                    query += f" AND type = ANY($3)"
                    params.append(type_values)
                
                # 添加其他过滤条件
                if filters:
                    for key, value in filters.items():
                        if key in ['source', 'target', 'conversation_id', 'session_id']:
                            params.append(value)
                            query += f" AND {key} = ${len(params)}"
                
                query += " ORDER BY timestamp ASC"
                
                # 执行查询
                rows = await conn.fetch(query, *params)
                
                for row in rows:
                    event = self._deserialize_event(dict(row))
                    if event:
                        events.append(event)
                
                self.stats["events_retrieved"] += len(events)
                
                logger.info(
                    "事件重播完成",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    event_count=len(events)
                )
                
            return events
            
        except Exception as e:
            logger.error("事件重播失败", error=str(e))
            return []
    
    def _deserialize_event(self, data: Dict[str, Any]) -> Optional[Event]:
        """反序列化事件"""
        try:
            # 处理时间戳
            if isinstance(data.get('timestamp'), str):
                timestamp = datetime.fromisoformat(data['timestamp'])
            else:
                timestamp = data.get('timestamp', utc_now())
            
            # 处理事件类型
            event_type = data.get('type', 'unknown')
            try:
                event_type = EventType(event_type)
            except ValueError:
                event_type = EventType.MESSAGE_SENT  # 默认类型
            
            # 处理优先级
            priority = data.get('priority', 'normal')
            try:
                priority = EventPriority(priority)
            except ValueError:
                priority = EventPriority.NORMAL
            
            # 创建事件对象
            event = Event(
                id=str(data.get('id', uuid.uuid4())),
                type=event_type,
                timestamp=timestamp,
                source=data.get('source', ''),
                target=data.get('target'),
                conversation_id=str(data['conversation_id']) if data.get('conversation_id') else None,
                session_id=str(data['session_id']) if data.get('session_id') else None,
                correlation_id=str(data['correlation_id']) if data.get('correlation_id') else None,
                priority=priority,
                data=json.loads(data['data']) if isinstance(data.get('data'), str) else data.get('data', {})
            )
            
            return event
            
        except Exception as e:
            logger.error("事件反序列化失败", error=str(e), data=data)
            return None
    
    async def create_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
        snapshot_data: Dict[str, Any],
        event_sequence: int
    ) -> str:
        """创建事件快照"""
        snapshot_id = str(uuid.uuid4())
        
        try:
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO event_snapshots (
                            id, aggregate_id, aggregate_type, snapshot_data, event_sequence
                        ) VALUES ($1, $2, $3, $4, $5)
                    """,
                        uuid.UUID(snapshot_id),
                        aggregate_id,
                        aggregate_type,
                        json.dumps(snapshot_data),
                        event_sequence
                    )
                    
                logger.info(
                    "事件快照创建成功",
                    snapshot_id=snapshot_id,
                    aggregate_id=aggregate_id,
                    event_sequence=event_sequence
                )
            
            return snapshot_id
            
        except Exception as e:
            logger.error("创建事件快照失败", error=str(e))
            raise
    
    async def get_latest_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        """获取最新的事件快照"""
        try:
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT snapshot_data, event_sequence FROM event_snapshots
                        WHERE aggregate_id = $1 AND aggregate_type = $2
                        ORDER BY event_sequence DESC
                        LIMIT 1
                    """, aggregate_id, aggregate_type)
                    
                    if row:
                        return json.loads(row['snapshot_data']), row['event_sequence']
            
            return None
            
        except Exception as e:
            logger.error("获取事件快照失败", error=str(e))
            return None
    
    async def move_to_dead_letter(
        self,
        event: Event,
        error_message: str,
        retry_count: int = 0
    ) -> None:
        """将失败事件移至死信队列"""
        try:
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO dead_letter_events (
                            id, original_event_id, event_data, error_message, retry_count, last_retry_at
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        uuid.uuid4(),
                        uuid.UUID(event.id) if hasattr(event, 'id') else uuid.uuid4(),
                        json.dumps(self._serialize_event(event)),
                        error_message,
                        retry_count,
                        utc_now()
                    )
                    
                logger.info("事件移至死信队列", event_id=event.id if hasattr(event, 'id') else None)
                
        except Exception as e:
            logger.error("移至死信队列失败", error=str(e))
    
    async def get_dead_letter_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取死信队列中的事件"""
        events = []
        
        try:
            if self.postgres:
                async with self.postgres.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT * FROM dead_letter_events
                        ORDER BY created_at DESC
                        LIMIT $1
                    """, limit)
                    
                    for row in rows:
                        events.append(dict(row))
            
            return events
            
        except Exception as e:
            logger.error("获取死信队列事件失败", error=str(e))
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return self.stats.copy()


class EventReplayService:
    """事件重播服务"""
    
    def __init__(self, event_store: EventStore, processing_engine: AsyncEventProcessingEngine):
        self.event_store = event_store
        self.processing_engine = processing_engine
        self.replaying = False
        self.replay_stats = {
            "total_replayed": 0,
            "successful": 0,
            "failed": 0,
            "last_replay_time": None
        }
    
    async def replay_for_agent(
        self,
        agent_id: str,
        from_time: datetime,
        to_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """为特定智能体重播事件"""
        if self.replaying:
            logger.warning("重播正在进行中")
            return {"status": "already_running"}
        
        self.replaying = True
        to_time = to_time or utc_now()
        
        try:
            # 获取需要重播的事件
            events = await self.event_store.replay_events(
                start_time=from_time,
                end_time=to_time,
                filters={"source": agent_id}
            )
            
            # 添加目标为该智能体的事件
            target_events = await self.event_store.replay_events(
                start_time=from_time,
                end_time=to_time,
                filters={"target": agent_id}
            )
            
            # 合并并去重
            all_events = self._merge_and_deduplicate(events, target_events)
            
            # 按时间排序
            all_events.sort(key=lambda e: e.timestamp if hasattr(e, 'timestamp') else utc_now())
            
            logger.info(
                "开始事件重播",
                agent_id=agent_id,
                event_count=len(all_events),
                from_time=from_time.isoformat(),
                to_time=to_time.isoformat()
            )
            
            # 重新处理事件
            for event in all_events:
                try:
                    # 添加重播标记
                    if hasattr(event, 'data') and isinstance(event.data, dict):
                        event.data['is_replay'] = True
                        event.data['original_timestamp'] = event.timestamp.isoformat() if hasattr(event, 'timestamp') else None
                    
                    # 使用高优先级处理重播事件
                    await self.processing_engine.submit_event(event, EventPriority.HIGH)
                    
                    self.replay_stats["successful"] += 1
                    
                except Exception as e:
                    logger.error("重播事件失败", event_id=event.id if hasattr(event, 'id') else None, error=str(e))
                    self.replay_stats["failed"] += 1
                
                self.replay_stats["total_replayed"] += 1
            
            self.replay_stats["last_replay_time"] = utc_now()
            
            result = {
                "status": "completed",
                "agent_id": agent_id,
                "events_replayed": len(all_events),
                "successful": self.replay_stats["successful"],
                "failed": self.replay_stats["failed"]
            }
            
            logger.info("事件重播完成", **result)
            
            return result
            
        except Exception as e:
            logger.error("事件重播异常", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
            
        finally:
            self.replaying = False
    
    async def replay_conversation(
        self,
        conversation_id: str,
        from_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """重播整个对话的事件"""
        if self.replaying:
            logger.warning("重播正在进行中")
            return {"status": "already_running"}
        
        self.replaying = True
        
        try:
            # 如果没有指定开始时间，使用很早的时间
            from_time = from_time or datetime(2020, 1, 1, tzinfo=timezone.utc)
            
            # 获取对话相关的所有事件
            events = await self.event_store.replay_events(
                start_time=from_time,
                end_time=utc_now(),
                filters={"conversation_id": conversation_id}
            )
            
            logger.info(
                "开始对话重播",
                conversation_id=conversation_id,
                event_count=len(events)
            )
            
            # 按时间顺序处理事件
            events.sort(key=lambda e: e.timestamp if hasattr(e, 'timestamp') else utc_now())
            
            for event in events:
                try:
                    # 标记为重播
                    if hasattr(event, 'data') and isinstance(event.data, dict):
                        event.data['is_replay'] = True
                    
                    await self.processing_engine.submit_event(event, EventPriority.NORMAL)
                    self.replay_stats["successful"] += 1
                    
                except Exception as e:
                    logger.error("重播对话事件失败", event_id=event.id if hasattr(event, 'id') else None, error=str(e))
                    self.replay_stats["failed"] += 1
                
                self.replay_stats["total_replayed"] += 1
            
            self.replay_stats["last_replay_time"] = utc_now()
            
            result = {
                "status": "completed",
                "conversation_id": conversation_id,
                "events_replayed": len(events),
                "successful": self.replay_stats["successful"],
                "failed": self.replay_stats["failed"]
            }
            
            logger.info("对话重播完成", **result)
            
            return result
            
        except Exception as e:
            logger.error("对话重播异常", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
            
        finally:
            self.replaying = False
    
    def _merge_and_deduplicate(self, *event_lists: List[Event]) -> List[Event]:
        """合并并去重事件列表"""
        seen_ids = set()
        merged = []
        
        for event_list in event_lists:
            for event in event_list:
                event_id = event.id if hasattr(event, 'id') else None
                if event_id and event_id not in seen_ids:
                    seen_ids.add(event_id)
                    merged.append(event)
        
        return merged
    
    async def replay_from_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str
    ) -> Dict[str, Any]:
        """从快照开始重播"""
        try:
            # 获取最新快照
            snapshot_result = await self.event_store.get_latest_snapshot(aggregate_id, aggregate_type)
            
            if not snapshot_result:
                logger.warning("未找到快照", aggregate_id=aggregate_id)
                return {
                    "status": "no_snapshot",
                    "aggregate_id": aggregate_id
                }
            
            snapshot_data, event_sequence = snapshot_result
            
            # 从快照之后的事件开始重播
            # 这里需要根据实际的事件序列实现
            logger.info(
                "从快照重播",
                aggregate_id=aggregate_id,
                event_sequence=event_sequence
            )
            
            return {
                "status": "completed",
                "aggregate_id": aggregate_id,
                "snapshot_sequence": event_sequence
            }
            
        except Exception as e:
            logger.error("从快照重播失败", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取重播统计信息"""
        return {
            **self.replay_stats,
            "is_replaying": self.replaying
        }