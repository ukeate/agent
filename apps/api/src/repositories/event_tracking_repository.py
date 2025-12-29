"""
事件追踪仓库 - 处理事件数据的数据库访问
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import asc, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.utils.timezone_utils import utc_now
from src.models.database.event_tracking import (
    EventDeduplication,
    EventError,
    EventSchema,
    EventStream,
)
from src.models.schemas.event_tracking import (
    CreateEventRequest,
    DataQuality,
    EventProcessingStats,
    EventStatus,
    EventType,

)

from src.core.logging import get_logger
logger = get_logger(__name__)

class EventStreamRepository:
    """事件流仓库"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_event(self, event: CreateEventRequest, data_quality: DataQuality) -> EventStream:
        """创建单个事件"""
        partition_key = event.event_timestamp.strftime("%Y-%m")

        db_event = EventStream(
            event_id=event.event_id,
            experiment_id=event.experiment_id,
            variant_id=event.variant_id,
            user_id=event.user_id,
            session_id=event.session_id,
            event_type=event.event_type.value,
            event_name=event.event_name,
            event_category=event.event_category,
            event_timestamp=event.event_timestamp,
            properties=event.properties,
            user_properties=event.user_properties,
            experiment_context=event.experiment_context,
            client_info=event.client_info.model_dump() if event.client_info else None,
            device_info=event.device_info.model_dump() if event.device_info else None,
            geo_info=event.geo_info.model_dump() if event.geo_info else None,
            status=EventStatus.PENDING.value,
            data_quality=data_quality.value,
            partition_key=partition_key,
        )

        self.db.add(db_event)
        try:
            await self.db.commit()
        except IntegrityError as e:
            await self.db.rollback()
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise ValueError(f"Event with id {event.event_id} already exists") from e
            raise
        except Exception:
            await self.db.rollback()
            raise

        await self.db.refresh(db_event)
        return db_event

    async def get_event_by_id(self, event_id: str) -> Optional[EventStream]:
        """根据事件ID获取事件"""
        result = await self.db.execute(select(EventStream).where(EventStream.event_id == event_id))
        return result.scalar_one_or_none()

    async def query_events(
        self,
        filters: Dict[str, Any],
        page: int,
        page_size: int,
        sort_by: str = "event_timestamp",
        sort_order: str = "desc",
    ) -> Tuple[List[EventStream], int]:
        """查询事件"""
        where_clauses = []

        if experiment_id := filters.get("experiment_id"):
            if isinstance(experiment_id, list):
                where_clauses.append(EventStream.experiment_id.in_(experiment_id))
            else:
                where_clauses.append(EventStream.experiment_id == experiment_id)

        if user_id := filters.get("user_id"):
            if isinstance(user_id, list):
                where_clauses.append(EventStream.user_id.in_(user_id))
            else:
                where_clauses.append(EventStream.user_id == user_id)

        if variant_id := filters.get("variant_id"):
            if isinstance(variant_id, list):
                where_clauses.append(EventStream.variant_id.in_(variant_id))
            else:
                where_clauses.append(EventStream.variant_id == variant_id)

        if event_type := filters.get("event_type"):
            if isinstance(event_type, list):
                where_clauses.append(EventStream.event_type.in_(event_type))
            else:
                where_clauses.append(EventStream.event_type == event_type)

        if event_name := filters.get("event_name"):
            if isinstance(event_name, list):
                where_clauses.append(EventStream.event_name.in_(event_name))
            else:
                where_clauses.append(EventStream.event_name == event_name)

        if start_time := filters.get("start_time"):
            where_clauses.append(EventStream.event_timestamp >= start_time)

        if end_time := filters.get("end_time"):
            where_clauses.append(EventStream.event_timestamp <= end_time)

        if status := filters.get("status"):
            if isinstance(status, list):
                where_clauses.append(EventStream.status.in_(status))
            else:
                where_clauses.append(EventStream.status == status)

        if data_quality := filters.get("data_quality"):
            if isinstance(data_quality, list):
                where_clauses.append(EventStream.data_quality.in_(data_quality))
            else:
                where_clauses.append(EventStream.data_quality == data_quality)

        total_stmt = select(func.count()).select_from(EventStream).where(*where_clauses)
        total_count = (await self.db.execute(total_stmt)).scalar_one()

        sort_columns = {
            "event_timestamp": EventStream.event_timestamp,
            "server_timestamp": EventStream.server_timestamp,
            "created_at": EventStream.created_at,
            "updated_at": EventStream.updated_at,
        }
        sort_column = sort_columns.get(sort_by, EventStream.event_timestamp)
        order_by = desc(sort_column) if sort_order.lower() == "desc" else asc(sort_column)

        stmt = (
            select(EventStream)
            .where(*where_clauses)
            .order_by(order_by)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        events = (await self.db.execute(stmt)).scalars().all()
        return events, total_count

    async def update_event_status(
        self,
        event_id: str,
        status: EventStatus,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新事件状态"""
        values: Dict[str, Any] = {"status": status.value}
        values["processed_at"] = utc_now() if status == EventStatus.PROCESSED else None
        if processing_metadata is not None:
            values["processing_metadata"] = processing_metadata

        result = await self.db.execute(
            update(EventStream).where(EventStream.event_id == event_id).values(**values)
        )
        await self.db.commit()
        return bool(result.rowcount)

    async def update_processed_at(self, event_id: str, processed_at: datetime) -> bool:
        """更新事件处理时间"""
        result = await self.db.execute(
            update(EventStream)
            .where(EventStream.event_id == event_id)
            .values(processed_at=processed_at)
        )
        await self.db.commit()
        return bool(result.rowcount)

    async def get_processing_stats(
        self,
        start_time: datetime,
        experiment_id: Optional[str] = None,
    ) -> EventProcessingStats:
        """获取事件处理统计"""
        where_clauses = [EventStream.created_at >= start_time]
        if experiment_id:
            where_clauses.append(EventStream.experiment_id == experiment_id)

        status_rows = (
            await self.db.execute(
                select(EventStream.status, func.count(EventStream.id))
                .where(*where_clauses)
                .group_by(EventStream.status)
            )
        ).all()
        events_by_status: Dict[EventStatus, int] = {}
        for status, count in status_rows:
            try:
                events_by_status[EventStatus(status)] = count
            except Exception:
                continue

        type_rows = (
            await self.db.execute(
                select(EventStream.event_type, func.count(EventStream.id))
                .where(*where_clauses)
                .group_by(EventStream.event_type)
            )
        ).all()
        events_by_type: Dict[EventType, int] = {}
        for event_type, count in type_rows:
            try:
                events_by_type[EventType(event_type)] = count
            except Exception:
                continue

        quality_rows = (
            await self.db.execute(
                select(EventStream.data_quality, func.count(EventStream.id))
                .where(*where_clauses)
                .group_by(EventStream.data_quality)
            )
        ).all()
        events_by_quality: Dict[DataQuality, int] = {}
        for quality, count in quality_rows:
            try:
                events_by_quality[DataQuality(quality)] = count
            except Exception:
                continue

        total_events = sum(events_by_status.values())

        newest_event = (
            await self.db.execute(
                select(EventStream.event_timestamp)
                .where(*where_clauses)
                .order_by(desc(EventStream.event_timestamp))
                .limit(1)
            )
        ).scalar_one_or_none()

        oldest_unprocessed_event = (
            await self.db.execute(
                select(EventStream.event_timestamp)
                .where(*where_clauses, EventStream.status == EventStatus.PENDING.value)
                .order_by(asc(EventStream.event_timestamp))
                .limit(1)
            )
        ).scalar_one_or_none()

        return EventProcessingStats(
            total_events=total_events,
            events_by_status=events_by_status,
            events_by_type=events_by_type,
            events_by_quality=events_by_quality,
            processing_rate_per_second=None,
            avg_processing_time_ms=None,
            error_rate_percentage=None,
            oldest_unprocessed_event=oldest_unprocessed_event,
            newest_event=newest_event,
            data_completeness_score=None,
            quality_score=None,
        )

class EventDeduplicationRepository:
    """事件去重仓库"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_duplicate(self, fingerprint: str) -> Optional[EventDeduplication]:
        result = await self.db.execute(
            select(EventDeduplication).where(EventDeduplication.event_fingerprint == fingerprint)
        )
        return result.scalar_one_or_none()

    async def record_event_fingerprint(
        self,
        fingerprint: str,
        original_event_id: str,
        experiment_id: str,
        user_id: str,
        event_timestamp: datetime,
        ttl_hours: int = 24,
    ) -> EventDeduplication:
        expires_at = utc_now() + timedelta(hours=ttl_hours)

        record = EventDeduplication(
            event_fingerprint=fingerprint,
            original_event_id=original_event_id,
            experiment_id=experiment_id,
            user_id=user_id,
            event_timestamp=event_timestamp,
            expires_at=expires_at,
        )
        self.db.add(record)

        try:
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            existing = await self.check_duplicate(fingerprint)
            if existing:
                return existing
            raise
        except Exception:
            await self.db.rollback()
            raise

        await self.db.refresh(record)
        return record

    async def update_duplicate_count(self, fingerprint: str) -> bool:
        result = await self.db.execute(
            update(EventDeduplication)
            .where(EventDeduplication.event_fingerprint == fingerprint)
            .values(
                duplicate_count=EventDeduplication.duplicate_count + 1,
                last_duplicate_at=utc_now(),
            )
        )
        await self.db.commit()
        return bool(result.rowcount)

class EventSchemaRepository:
    """事件Schema仓库"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_active_schema_by_event(
        self,
        event_type: str,
        event_name: str,
        schema_version: Optional[str] = None,
    ) -> Optional[EventSchema]:
        stmt = select(EventSchema).where(
            EventSchema.is_active.is_(True),
            EventSchema.event_type == event_type,
            EventSchema.event_name == event_name,
        )
        if schema_version:
            stmt = stmt.where(EventSchema.schema_version == schema_version)
        else:
            stmt = stmt.order_by(desc(EventSchema.schema_version))

        result = await self.db.execute(stmt.limit(1))
        return result.scalar_one_or_none()

class EventErrorRepository:
    """事件错误仓库"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_error(
        self,
        failed_event_id: Optional[str],
        raw_event_data: Dict[str, Any],
        error_type: str,
        error_message: str,
        processing_stage: str,
        error_details: Optional[Dict[str, Any]] = None,
        is_recoverable: Optional[bool] = None,
    ) -> EventError:
        record = EventError(
            failed_event_id=failed_event_id,
            raw_event_data=raw_event_data,
            error_type=error_type,
            error_message=error_message,
            error_details=error_details,
            processing_stage=processing_stage,
            is_recoverable=is_recoverable,
        )

        self.db.add(record)
        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise

        await self.db.refresh(record)
        logger.error("Recorded event error", error_type=error_type, stage=processing_stage)
        return record
