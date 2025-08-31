"""
事件追踪仓库 - 处理事件数据的数据库访问
"""
import hashlib
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc, text, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from models.database.event_tracking import (
    EventStream, EventAggregation, EventMetric, EventBatch, 
    EventSchema, EventDeduplication, EventError,
    EventType, EventStatus, DataQuality
)
from models.schemas.event_tracking import (
    CreateEventRequest, BatchEventsRequest, EventQueryRequest,
    EventDeduplicationInfo, EventValidationResult
)
from core.logging import logger


class EventStreamRepository:
    """事件流仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        self.model = EventStream
    
    def create_event(self, event_request: CreateEventRequest) -> EventStream:
        """创建单个事件"""
        try:
            # 生成分区键
            partition_key = event_request.event_timestamp.strftime('%Y-%m') if event_request.event_timestamp else utc_now().strftime('%Y-%m')
            
            # 创建事件对象
            event = EventStream(
                event_id=event_request.event_id,
                experiment_id=event_request.experiment_id,
                variant_id=event_request.variant_id,
                user_id=event_request.user_id,
                session_id=event_request.session_id,
                
                event_type=event_request.event_type.value,
                event_name=event_request.event_name,
                event_category=event_request.event_category,
                
                event_timestamp=event_request.event_timestamp,
                
                properties=event_request.properties,
                user_properties=event_request.user_properties,
                experiment_context=event_request.experiment_context,
                
                client_info=event_request.client_info.dict() if event_request.client_info else None,
                device_info=event_request.device_info.dict() if event_request.device_info else None,
                geo_info=event_request.geo_info.dict() if event_request.geo_info else None,
                
                status=EventStatus.PENDING.value,
                data_quality=event_request.data_quality.value if event_request.data_quality else DataQuality.HIGH.value,
                
                partition_key=partition_key
            )
            
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            
            logger.debug(f"Created event {event.event_id} for experiment {event.experiment_id}")
            return event
            
        except IntegrityError as e:
            self.db.rollback()
            if "duplicate key" in str(e).lower():
                logger.warning(f"Duplicate event_id: {event_request.event_id}")
                raise ValueError(f"Event with ID {event_request.event_id} already exists")
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating event: {str(e)}")
            raise
    
    def batch_create_events(self, events_request: BatchEventsRequest) -> Tuple[List[EventStream], List[str]]:
        """批量创建事件"""
        created_events = []
        failed_events = []
        
        try:
            for event_request in events_request.events:
                try:
                    event = self.create_event(event_request)
                    created_events.append(event)
                except Exception as e:
                    failed_events.append(f"Event {event_request.event_id}: {str(e)}")
                    logger.error(f"Failed to create event {event_request.event_id}: {str(e)}")
                    continue
            
            logger.info(f"Batch created {len(created_events)} events, {len(failed_events)} failed")
            return created_events, failed_events
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in batch create events: {str(e)}")
            raise
    
    def get_events(self, query_request: EventQueryRequest) -> Tuple[List[EventStream], int]:
        """查询事件"""
        try:
            query = self.db.query(self.model)
            
            # 应用过滤条件
            if query_request.experiment_ids:
                query = query.filter(self.model.experiment_id.in_(query_request.experiment_ids))
            
            if query_request.user_ids:
                query = query.filter(self.model.user_id.in_(query_request.user_ids))
            
            if query_request.variant_ids:
                query = query.filter(self.model.variant_id.in_(query_request.variant_ids))
            
            if query_request.event_types:
                event_type_values = [et.value for et in query_request.event_types]
                query = query.filter(self.model.event_type.in_(event_type_values))
            
            if query_request.event_names:
                query = query.filter(self.model.event_name.in_(query_request.event_names))
            
            if query_request.start_time:
                query = query.filter(self.model.event_timestamp >= query_request.start_time)
            
            if query_request.end_time:
                query = query.filter(self.model.event_timestamp <= query_request.end_time)
            
            if query_request.status:
                status_values = [s.value for s in query_request.status]
                query = query.filter(self.model.status.in_(status_values))
            
            if query_request.data_quality:
                quality_values = [q.value for q in query_request.data_quality]
                query = query.filter(self.model.data_quality.in_(quality_values))
            
            # 获取总数
            total_count = query.count()
            
            # 应用排序
            if query_request.sort_by:
                sort_column = getattr(self.model, query_request.sort_by, None)
                if sort_column:
                    if query_request.sort_order == 'desc':
                        query = query.order_by(desc(sort_column))
                    else:
                        query = query.order_by(asc(sort_column))
            
            # 应用分页
            offset = (query_request.page - 1) * query_request.page_size
            events = query.offset(offset).limit(query_request.page_size).all()
            
            return events, total_count
            
        except Exception as e:
            logger.error(f"Error querying events: {str(e)}")
            raise
    
    def update_event_status(self, event_id: str, status: EventStatus, 
                          processing_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新事件状态"""
        try:
            update_data = {
                'status': status.value,
                'processed_at': utc_now() if status == EventStatus.PROCESSED else None
            }
            
            if processing_metadata:
                update_data['processing_metadata'] = processing_metadata
            
            result = self.db.query(self.model).filter(
                self.model.event_id == event_id
            ).update(update_data)
            
            self.db.commit()
            
            if result > 0:
                logger.debug(f"Updated event {event_id} status to {status.value}")
                return True
            else:
                logger.warning(f"Event {event_id} not found for status update")
                return False
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating event status: {str(e)}")
            raise
    
    def get_events_by_experiment_and_period(self, experiment_id: str, 
                                           start_time: datetime, end_time: datetime,
                                           event_types: Optional[List[str]] = None) -> List[EventStream]:
        """获取实验在指定时间段的事件"""
        try:
            query = self.db.query(self.model).filter(
                and_(
                    self.model.experiment_id == experiment_id,
                    self.model.event_timestamp >= start_time,
                    self.model.event_timestamp <= end_time,
                    self.model.status == EventStatus.PROCESSED.value
                )
            )
            
            if event_types:
                query = query.filter(self.model.event_type.in_(event_types))
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error getting events by experiment and period: {str(e)}")
            raise
    
    async def get_event_by_id(self, event_id: str) -> Optional[EventStream]:
        """根据事件ID获取事件"""
        try:
            return self.db.query(self.model).filter(
                self.model.event_id == event_id
            ).first()
        except Exception as e:
            logger.error(f"Error getting event by id: {str(e)}")
            raise

    async def get_event_by_db_id(self, db_id: str) -> Optional[EventStream]:
        """根据数据库ID获取事件"""
        try:
            return self.db.query(self.model).filter(
                self.model.id == db_id
            ).first()
        except Exception as e:
            logger.error(f"Error getting event by db id: {str(e)}")
            raise

    async def update_processed_at(self, event_id: str, processed_at: datetime) -> bool:
        """更新事件处理时间"""
        try:
            result = self.db.query(self.model).filter(
                self.model.event_id == event_id
            ).update({
                'processed_at': processed_at
            })
            self.db.commit()
            return result > 0
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating processed_at: {str(e)}")
            raise

    async def create_event(self, event: CreateEventRequest, data_quality: DataQuality) -> EventStream:
        """创建单个事件（异步版本）"""
        try:
            # 生成分区键
            partition_key = event.event_timestamp.strftime('%Y-%m') if event.event_timestamp else utc_now().strftime('%Y-%m')
            
            # 创建事件对象
            db_event = self.model(
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
                
                client_info=event.client_info.dict() if event.client_info else None,
                device_info=event.device_info.dict() if event.device_info else None,
                geo_info=event.geo_info.dict() if event.geo_info else None,
                
                data_quality=data_quality.value,
                partition_key=partition_key
            )
            
            self.db.add(db_event)
            self.db.commit()
            self.db.refresh(db_event)
            
            return db_event
            
        except IntegrityError as e:
            self.db.rollback()
            if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                raise ValueError(f"Event with id {event.event_id} already exists")
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating event: {str(e)}")
            raise

    async def query_events(self, filters: Dict[str, Any], page: int, page_size: int, 
                          sort_by: str = 'event_timestamp', sort_order: str = 'desc') -> Tuple[List[EventStream], int]:
        """查询事件（异步版本）"""
        try:
            query = self.db.query(self.model)
            
            # 应用过滤条件
            if 'experiment_id' in filters:
                if isinstance(filters['experiment_id'], list):
                    query = query.filter(self.model.experiment_id.in_(filters['experiment_id']))
                else:
                    query = query.filter(self.model.experiment_id == filters['experiment_id'])
            
            if 'user_id' in filters:
                if isinstance(filters['user_id'], list):
                    query = query.filter(self.model.user_id.in_(filters['user_id']))
                else:
                    query = query.filter(self.model.user_id == filters['user_id'])
            
            if 'variant_id' in filters:
                if isinstance(filters['variant_id'], list):
                    query = query.filter(self.model.variant_id.in_(filters['variant_id']))
                else:
                    query = query.filter(self.model.variant_id == filters['variant_id'])
            
            if 'event_type' in filters:
                if isinstance(filters['event_type'], list):
                    query = query.filter(self.model.event_type.in_(filters['event_type']))
                else:
                    query = query.filter(self.model.event_type == filters['event_type'])
            
            if 'event_name' in filters:
                if isinstance(filters['event_name'], list):
                    query = query.filter(self.model.event_name.in_(filters['event_name']))
                else:
                    query = query.filter(self.model.event_name == filters['event_name'])
            
            if 'start_time' in filters:
                query = query.filter(self.model.event_timestamp >= filters['start_time'])
            
            if 'end_time' in filters:
                query = query.filter(self.model.event_timestamp <= filters['end_time'])
            
            if 'status' in filters:
                if isinstance(filters['status'], list):
                    query = query.filter(self.model.status.in_(filters['status']))
                else:
                    query = query.filter(self.model.status == filters['status'])
            
            if 'data_quality' in filters:
                if isinstance(filters['data_quality'], list):
                    query = query.filter(self.model.data_quality.in_(filters['data_quality']))
                else:
                    query = query.filter(self.model.data_quality == filters['data_quality'])
            
            # 获取总数
            total_count = query.count()
            
            # 排序
            if sort_order.lower() == 'desc':
                query = query.order_by(desc(getattr(self.model, sort_by, self.model.event_timestamp)))
            else:
                query = query.order_by(asc(getattr(self.model, sort_by, self.model.event_timestamp)))
            
            # 分页
            query = query.offset((page - 1) * page_size).limit(page_size)
            
            events = query.all()
            return events, total_count
            
        except Exception as e:
            logger.error(f"Error querying events: {str(e)}")
            raise

    async def get_processing_stats(self, start_time: Optional[datetime] = None, 
                                  experiment_id: Optional[str] = None) -> Any:
        """获取处理统计信息（异步版本）"""
        try:
            from models.schemas.event_tracking import EventProcessingStats
            
            query = self.db.query(self.model)
            
            if start_time:
                query = query.filter(self.model.created_at >= start_time)
            if experiment_id:
                query = query.filter(self.model.experiment_id == experiment_id)
            
            # 按状态统计
            status_stats = {}
            status_results = self.db.query(
                self.model.status,
                func.count(self.model.id).label('count')
            )
            if start_time:
                status_results = status_results.filter(self.model.created_at >= start_time)
            if experiment_id:
                status_results = status_results.filter(self.model.experiment_id == experiment_id)
            
            status_results = status_results.group_by(self.model.status).all()
            
            for status, count in status_results:
                status_stats[EventStatus(status)] = count
            
            # 按类型统计
            type_stats = {}
            type_results = self.db.query(
                self.model.event_type,
                func.count(self.model.id).label('count')
            )
            if start_time:
                type_results = type_results.filter(self.model.created_at >= start_time)
            if experiment_id:
                type_results = type_results.filter(self.model.experiment_id == experiment_id)
            
            type_results = type_results.group_by(self.model.event_type).all()
            
            for event_type, count in type_results:
                type_stats[EventType(event_type)] = count
            
            # 按质量统计
            quality_stats = {}
            quality_results = self.db.query(
                self.model.data_quality,
                func.count(self.model.id).label('count')
            )
            if start_time:
                quality_results = quality_results.filter(self.model.created_at >= start_time)
            if experiment_id:
                quality_results = quality_results.filter(self.model.experiment_id == experiment_id)
            
            quality_results = quality_results.group_by(self.model.data_quality).all()
            
            for quality, count in quality_results:
                quality_stats[DataQuality(quality)] = count
            
            # 总事件数
            total_events = sum(status_stats.values())
            
            # 最新和最旧事件
            newest_event = query.order_by(desc(self.model.event_timestamp)).first()
            oldest_unprocessed = query.filter(
                self.model.status == EventStatus.PENDING.value
            ).order_by(asc(self.model.event_timestamp)).first()
            
            return EventProcessingStats(
                total_events=total_events,
                events_by_status=status_stats,
                events_by_type=type_stats,
                events_by_quality=quality_stats,
                newest_event=newest_event.event_timestamp if newest_event else None,
                oldest_unprocessed_event=oldest_unprocessed.event_timestamp if oldest_unprocessed else None
            )
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            raise

    def get_processing_statistics(self, start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            query = self.db.query(self.model)
            
            if start_time:
                query = query.filter(self.model.created_at >= start_time)
            if end_time:
                query = query.filter(self.model.created_at <= end_time)
            
            # 按状态统计
            status_stats = self.db.query(
                self.model.status,
                func.count(self.model.id).label('count')
            ).filter(
                query.whereclause if hasattr(query, 'whereclause') else True
            ).group_by(self.model.status).all()
            
            # 按类型统计
            type_stats = self.db.query(
                self.model.event_type,
                func.count(self.model.id).label('count')
            ).filter(
                query.whereclause if hasattr(query, 'whereclause') else True
            ).group_by(self.model.event_type).all()
            
            # 按质量统计
            quality_stats = self.db.query(
                self.model.data_quality,
                func.count(self.model.id).label('count')
            ).filter(
                query.whereclause if hasattr(query, 'whereclause') else True
            ).group_by(self.model.data_quality).all()
            
            # 总数
            total_count = sum(count for _, count in status_stats)
            
            return {
                'total_events': total_count,
                'status_stats': dict(status_stats),
                'type_stats': dict(type_stats),
                'quality_stats': dict(quality_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {str(e)}")
            raise


class EventDeduplicationRepository:
    """事件去重仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        self.model = EventDeduplication
    
    def generate_event_fingerprint(self, event_request: CreateEventRequest) -> str:
        """生成事件指纹"""
        try:
            # 使用关键字段生成指纹
            fingerprint_data = {
                'user_id': event_request.user_id,
                'experiment_id': event_request.experiment_id,
                'event_type': event_request.event_type.value,
                'event_name': event_request.event_name,
                'event_timestamp': event_request.event_timestamp.isoformat() if event_request.event_timestamp else None
            }
            
            # 添加重要的属性到指纹中（可根据需要调整）
            if event_request.properties:
                # 只包含关键属性
                key_properties = {k: v for k, v in event_request.properties.items() 
                                if k in ['transaction_id', 'order_id', 'product_id']}  # 可配置
                if key_properties:
                    fingerprint_data['key_properties'] = key_properties
            
            # 生成SHA-256哈希
            fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=True)
            fingerprint = hashlib.sha256(fingerprint_json.encode('utf-8')).hexdigest()
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error generating event fingerprint: {str(e)}")
            raise
    
    def check_duplicate(self, event_request: CreateEventRequest, 
                       ttl_hours: int = 24) -> EventDeduplicationInfo:
        """检查事件是否重复"""
        try:
            fingerprint = self.generate_event_fingerprint(event_request)
            
            # 查询是否存在
            existing = self.db.query(self.model).filter(
                self.model.event_fingerprint == fingerprint
            ).first()
            
            if existing:
                # 更新重复计数
                existing.duplicate_count += 1
                existing.last_duplicate_at = utc_now()
                self.db.commit()
                
                return EventDeduplicationInfo(
                    event_fingerprint=fingerprint,
                    is_duplicate=True,
                    original_event_id=existing.original_event_id,
                    duplicate_count=existing.duplicate_count,
                    first_seen_at=existing.first_seen_at,
                    last_duplicate_at=existing.last_duplicate_at
                )
            else:
                # 创建新的去重记录
                expires_at = utc_now() + timedelta(hours=ttl_hours)
                
                dedup_record = self.model(
                    event_fingerprint=fingerprint,
                    original_event_id=event_request.event_id,
                    experiment_id=event_request.experiment_id,
                    user_id=event_request.user_id,
                    event_timestamp=event_request.event_timestamp,
                    expires_at=expires_at
                )
                
                self.db.add(dedup_record)
                self.db.commit()
                
                return EventDeduplicationInfo(
                    event_fingerprint=fingerprint,
                    is_duplicate=False,
                    original_event_id=event_request.event_id,
                    duplicate_count=0,
                    first_seen_at=utc_now()
                )
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error checking duplicate: {str(e)}")
            raise
    
    async def check_duplicate(self, fingerprint: str) -> Optional[EventDeduplication]:
        """检查指纹是否重复（异步版本）"""
        try:
            return self.db.query(self.model).filter(
                self.model.event_fingerprint == fingerprint
            ).first()
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            raise

    async def record_event_fingerprint(self, fingerprint: str, original_event_id: str,
                                     experiment_id: str, user_id: str, 
                                     event_timestamp: datetime, ttl_hours: int = 24) -> EventDeduplication:
        """记录事件指纹（异步版本）"""
        try:
            expires_at = utc_now() + timedelta(hours=ttl_hours)
            
            dedup_record = self.model(
                event_fingerprint=fingerprint,
                original_event_id=original_event_id,
                experiment_id=experiment_id,
                user_id=user_id,
                event_timestamp=event_timestamp,
                expires_at=expires_at
            )
            
            self.db.add(dedup_record)
            self.db.commit()
            self.db.refresh(dedup_record)
            
            return dedup_record
            
        except IntegrityError:
            self.db.rollback()
            # 如果已存在，返回现有记录
            return self.db.query(self.model).filter(
                self.model.event_fingerprint == fingerprint
            ).first()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording event fingerprint: {str(e)}")
            raise

    async def update_duplicate_count(self, fingerprint: str) -> bool:
        """更新重复计数（异步版本）"""
        try:
            result = self.db.query(self.model).filter(
                self.model.event_fingerprint == fingerprint
            ).update({
                'duplicate_count': self.model.duplicate_count + 1,
                'last_duplicate_at': utc_now()
            })
            
            self.db.commit()
            return result > 0
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating duplicate count: {str(e)}")
            raise

    def cleanup_expired_records(self) -> int:
        """清理过期的去重记录"""
        try:
            current_time = utc_now()
            result = self.db.query(self.model).filter(
                self.model.expires_at <= current_time
            ).delete()
            
            self.db.commit()
            
            if result > 0:
                logger.info(f"Cleaned up {result} expired deduplication records")
            
            return result
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error cleaning up expired records: {str(e)}")
            raise


class EventSchemaRepository:
    """事件Schema仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        self.model = EventSchema
    
    def create_schema(self, schema_data: Dict[str, Any]) -> EventSchema:
        """创建事件Schema"""
        try:
            schema = self.model(**schema_data)
            self.db.add(schema)
            self.db.commit()
            self.db.refresh(schema)
            
            logger.info(f"Created event schema {schema.schema_name} v{schema.schema_version}")
            return schema
            
        except IntegrityError as e:
            self.db.rollback()
            if "duplicate key" in str(e).lower():
                raise ValueError(f"Schema {schema_data['schema_name']} v{schema_data['schema_version']} already exists")
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating event schema: {str(e)}")
            raise
    
    async def get_active_schema_by_event(self, event_type: str, event_name: str,
                                       schema_version: Optional[str] = None) -> Optional[EventSchema]:
        """获取事件Schema（异步版本）"""
        try:
            query = self.db.query(self.model).filter(
                and_(
                    self.model.event_type == event_type,
                    self.model.event_name == event_name,
                    self.model.is_active == True
                )
            )
            
            if schema_version:
                query = query.filter(self.model.schema_version == schema_version)
            else:
                # 获取最新版本
                query = query.order_by(desc(self.model.schema_version))
            
            return query.first()
            
        except Exception as e:
            logger.error(f"Error getting event schema: {str(e)}")
            raise

    def get_schema(self, event_type: str, event_name: str, 
                  schema_version: Optional[str] = None) -> Optional[EventSchema]:
        """获取事件Schema"""
        try:
            query = self.db.query(self.model).filter(
                and_(
                    self.model.event_type == event_type,
                    self.model.event_name == event_name,
                    self.model.is_active == True
                )
            )
            
            if schema_version:
                query = query.filter(self.model.schema_version == schema_version)
            else:
                # 获取最新版本
                query = query.order_by(desc(self.model.schema_version))
            
            return query.first()
            
        except Exception as e:
            logger.error(f"Error getting event schema: {str(e)}")
            raise
    
    def update_validation_stats(self, schema_id: str, success: bool) -> None:
        """更新验证统计"""
        try:
            schema = self.db.query(self.model).filter(self.model.id == schema_id).first()
            if schema:
                schema.events_validated += 1
                schema.last_validation_at = utc_now()
                
                # 计算成功率（简单移动平均）
                if schema.validation_success_rate is None:
                    schema.validation_success_rate = 1.0 if success else 0.0
                else:
                    # 指数移动平均
                    alpha = 0.1
                    new_value = 1.0 if success else 0.0
                    schema.validation_success_rate = (
                        alpha * new_value + 
                        (1 - alpha) * schema.validation_success_rate
                    )
                
                self.db.commit()
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating validation stats: {str(e)}")


class EventErrorRepository:
    """事件错误仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        self.model = EventError
    
    async def create_error(self, failed_event_id: Optional[str], raw_event_data: Dict[str, Any],
                          error_type: str, error_message: str, processing_stage: str,
                          error_details: Optional[Dict[str, Any]] = None,
                          is_recoverable: Optional[bool] = None) -> EventError:
        """记录事件处理错误（异步版本）"""
        try:
            error_record = self.model(
                failed_event_id=failed_event_id,
                raw_event_data=raw_event_data,
                error_type=error_type,
                error_message=error_message,
                error_details=error_details,
                processing_stage=processing_stage,
                is_recoverable=is_recoverable
            )
            
            self.db.add(error_record)
            self.db.commit()
            self.db.refresh(error_record)
            
            logger.error(f"Recorded event error: {error_type} at {processing_stage}")
            return error_record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording event error: {str(e)}")
            raise

    def record_error(self, failed_event_id: Optional[str], raw_event_data: Dict[str, Any],
                    error_type: str, error_message: str, processing_stage: str,
                    error_details: Optional[Dict[str, Any]] = None,
                    is_recoverable: Optional[bool] = None) -> EventError:
        """记录事件处理错误"""
        try:
            error_record = self.model(
                failed_event_id=failed_event_id,
                raw_event_data=raw_event_data,
                error_type=error_type,
                error_message=error_message,
                error_details=error_details,
                processing_stage=processing_stage,
                is_recoverable=is_recoverable
            )
            
            self.db.add(error_record)
            self.db.commit()
            self.db.refresh(error_record)
            
            logger.error(f"Recorded event error: {error_type} at {processing_stage}")
            return error_record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording event error: {str(e)}")
            raise
    
    def get_recoverable_errors(self, limit: int = 100) -> List[EventError]:
        """获取可恢复的错误"""
        try:
            return self.db.query(self.model).filter(
                and_(
                    self.model.is_recoverable == True,
                    self.model.resolved_at.is_(None),
                    self.model.retry_count < 5  # 最大重试次数
                )
            ).order_by(asc(self.model.occurred_at)).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error getting recoverable errors: {str(e)}")
            raise
    
    def mark_error_resolved(self, error_id: str, resolution_method: str) -> bool:
        """标记错误已解决"""
        try:
            result = self.db.query(self.model).filter(
                self.model.id == error_id
            ).update({
                'resolved_at': utc_now(),
                'resolution_method': resolution_method
            })
            
            self.db.commit()
            return result > 0
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error marking error resolved: {str(e)}")
            raise