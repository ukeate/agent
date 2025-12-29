"""
反馈数据访问层

提供反馈数据的CRUD操作和查询功能
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_, desc, func, select, update, case, cast, Float
from sqlalchemy.exc import SQLAlchemyError
from src.core.utils.timezone_utils import utc_now
from ..models.schemas.feedback import (
    FeedbackEvent, FeedbackBatch, UserFeedbackProfile,
    ItemFeedbackSummary, RewardSignal, FeedbackQualityLog,
    FeedbackAggregation,
)

from src.core.logging import get_logger
logger = get_logger(__name__)

class FeedbackRepository:
    """反馈数据仓库"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_feedback_event(self, event_data: Dict) -> FeedbackEvent:
        """创建反馈事件"""
        try:
            event = FeedbackEvent(**event_data)
            self.db.add(event)
            await self.db.commit()
            await self.db.refresh(event)
            return event
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"创建反馈事件失败: {e}")
            raise

    async def create_feedback_batch(self, events: List[Dict], batch_data: Dict) -> FeedbackBatch:
        """批量创建反馈事件"""
        try:
            batch = FeedbackBatch(**batch_data)
            self.db.add(batch)
            await self.db.flush()

            event_objects: List[FeedbackEvent] = []
            for event_data in events:
                event_data['batch_id'] = batch.id
                event_objects.append(FeedbackEvent(**event_data))

            self.db.add_all(event_objects)
            batch.event_count = len(event_objects)

            await self.db.commit()
            await self.db.refresh(batch)
            return batch
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"批量创建反馈事件失败: {e}")
            raise

    async def get_feedback_events(
        self,
        user_id: str = None,
        item_id: str = None,
        feedback_types: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FeedbackEvent]:
        """查询反馈事件"""
        try:
            stmt = select(FeedbackEvent).where(FeedbackEvent.valid.is_(True))

            if user_id:
                stmt = stmt.where(FeedbackEvent.user_id == user_id)
            if item_id:
                stmt = stmt.where(FeedbackEvent.item_id == item_id)
            if feedback_types:
                stmt = stmt.where(FeedbackEvent.feedback_type.in_(feedback_types))
            if start_date:
                stmt = stmt.where(FeedbackEvent.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(FeedbackEvent.timestamp <= end_date)

            stmt = stmt.order_by(desc(FeedbackEvent.timestamp)).offset(offset).limit(limit)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"查询反馈事件失败: {e}")
            raise

    async def update_feedback_processing_status(self, event_ids: List[str], processed: bool = True) -> None:
        """更新反馈处理状态"""
        try:
            await self.db.execute(
                update(FeedbackEvent)
                .where(FeedbackEvent.event_id.in_(event_ids))
                .values(
                    processed=processed,
                    processed_at=utc_now() if processed else None,
                )
            )
            await self.db.commit()
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"更新反馈处理状态失败: {e}")
            raise

    async def get_or_create_user_profile(self, user_id: str) -> UserFeedbackProfile:
        """获取或创建用户反馈档案"""
        try:
            result = await self.db.execute(
                select(UserFeedbackProfile).where(UserFeedbackProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if not profile:
                profile = UserFeedbackProfile(user_id=user_id)
                self.db.add(profile)
                await self.db.commit()
                await self.db.refresh(profile)

            return profile
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"获取或创建用户档案失败: {e}")
            raise

    async def update_user_profile(self, user_id: str, profile_data: Dict) -> None:
        """更新用户反馈档案"""
        try:
            await self.db.execute(
                update(UserFeedbackProfile)
                .where(UserFeedbackProfile.user_id == user_id)
                .values(**profile_data)
            )
            await self.db.commit()
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"更新用户档案失败: {e}")
            raise

    async def get_user_feedback_stats(self, user_id: str) -> Dict:
        """获取用户反馈统计"""
        try:
            stats_stmt = select(
                func.count(FeedbackEvent.id).label('total_count'),
                func.count(func.distinct(FeedbackEvent.item_id)).label('unique_items'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality'),
                func.min(FeedbackEvent.timestamp).label('first_feedback'),
                func.max(FeedbackEvent.timestamp).label('last_feedback'),
            ).where(
                FeedbackEvent.user_id == user_id,
                FeedbackEvent.valid.is_(True),
            )
            stats = (await self.db.execute(stats_stmt)).one()

            type_stmt = select(
                FeedbackEvent.feedback_type,
                func.count(FeedbackEvent.id).label('count'),
            ).where(
                FeedbackEvent.user_id == user_id,
                FeedbackEvent.valid.is_(True),
            ).group_by(FeedbackEvent.feedback_type)
            type_distribution = (await self.db.execute(type_stmt)).all()

            return {
                'total_feedbacks': stats.total_count or 0,
                'unique_items': stats.unique_items or 0,
                'average_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
                'first_feedback_time': stats.first_feedback,
                'last_feedback_time': stats.last_feedback,
                'type_distribution': {t.feedback_type: t.count for t in type_distribution},
            }
        except SQLAlchemyError as e:
            logger.error(f"获取用户反馈统计失败: {e}")
            raise

    async def get_or_create_item_summary(self, item_id: str) -> ItemFeedbackSummary:
        """获取或创建推荐项汇总"""
        try:
            result = await self.db.execute(
                select(ItemFeedbackSummary).where(ItemFeedbackSummary.item_id == item_id)
            )
            summary = result.scalar_one_or_none()

            if not summary:
                summary = ItemFeedbackSummary(item_id=item_id)
                self.db.add(summary)
                await self.db.commit()
                await self.db.refresh(summary)

            return summary
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"获取或创建推荐项汇总失败: {e}")
            raise

    async def update_item_summary(self, item_id: str, summary_data: Dict) -> None:
        """更新推荐项汇总"""
        try:
            await self.db.execute(
                update(ItemFeedbackSummary)
                .where(ItemFeedbackSummary.item_id == item_id)
                .values(**summary_data)
            )
            await self.db.commit()
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"更新推荐项汇总失败: {e}")
            raise

    async def get_item_feedback_stats(self, item_id: str) -> Dict:
        """获取推荐项反馈统计"""
        try:
            stats_stmt = select(
                func.count(FeedbackEvent.id).label('total_count'),
                func.count(func.distinct(FeedbackEvent.user_id)).label('unique_users'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality'),
                func.min(FeedbackEvent.timestamp).label('first_feedback'),
                func.max(FeedbackEvent.timestamp).label('last_feedback'),
            ).where(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.valid.is_(True),
            )
            stats = (await self.db.execute(stats_stmt)).one()

            rating_stmt = select(
                func.avg(cast(FeedbackEvent.value, Float)).label('avg_rating'),
                func.count(FeedbackEvent.id).label('rating_count'),
            ).where(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.feedback_type == 'rating',
                FeedbackEvent.valid.is_(True),
            )
            rating_stats = (await self.db.execute(rating_stmt)).one()

            like_stmt = select(
                func.count(case((FeedbackEvent.feedback_type == 'like', 1))).label('like_count'),
                func.count(case((FeedbackEvent.feedback_type == 'dislike', 1))).label('dislike_count'),
            ).where(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.feedback_type.in_(['like', 'dislike']),
                FeedbackEvent.valid.is_(True),
            )
            like_stats = (await self.db.execute(like_stmt)).one()

            total_likes = (like_stats.like_count or 0) + (like_stats.dislike_count or 0)
            like_ratio = (like_stats.like_count or 0) / total_likes if total_likes > 0 else 0.0

            return {
                'total_feedbacks': stats.total_count or 0,
                'unique_users': stats.unique_users or 0,
                'average_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
                'first_feedback_time': stats.first_feedback,
                'last_feedback_time': stats.last_feedback,
                'average_rating': float(rating_stats.avg_rating) if rating_stats.avg_rating else None,
                'rating_count': rating_stats.rating_count or 0,
                'like_count': like_stats.like_count or 0,
                'dislike_count': like_stats.dislike_count or 0,
                'like_ratio': like_ratio,
            }
        except SQLAlchemyError as e:
            logger.error(f"获取推荐项反馈统计失败: {e}")
            raise

    async def save_reward_signal(self, reward_data: Dict) -> RewardSignal:
        """保存奖励信号"""
        try:
            existing_result = await self.db.execute(
                select(RewardSignal)
                .where(
                    RewardSignal.user_id == reward_data['user_id'],
                    RewardSignal.item_id == reward_data['item_id'],
                )
                .order_by(desc(RewardSignal.calculated_at))
                .limit(1)
            )
            existing = existing_result.scalar_one_or_none()

            if existing and existing.valid_until and existing.valid_until > utc_now():
                for key, value in reward_data.items():
                    setattr(existing, key, value)
                reward = existing
            else:
                reward = RewardSignal(**reward_data)
                self.db.add(reward)

            await self.db.commit()
            await self.db.refresh(reward)
            return reward
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存奖励信号失败: {e}")
            raise

    async def get_latest_reward_signal(self, user_id: str, item_id: str) -> Optional[RewardSignal]:
        """获取最新的奖励信号"""
        try:
            result = await self.db.execute(
                select(RewardSignal)
                .where(
                    RewardSignal.user_id == user_id,
                    RewardSignal.item_id == item_id,
                    or_(
                        RewardSignal.valid_until.is_(None),
                        RewardSignal.valid_until > utc_now(),
                    ),
                )
                .order_by(desc(RewardSignal.calculated_at))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"获取最新奖励信号失败: {e}")
            raise

    async def save_quality_assessment(self, quality_data: Dict) -> FeedbackQualityLog:
        """保存质量评估结果"""
        try:
            quality_log = FeedbackQualityLog(**quality_data)
            self.db.add(quality_log)
            await self.db.commit()
            await self.db.refresh(quality_log)
            return quality_log
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存质量评估失败: {e}")
            raise

    async def get_or_create_aggregation(
        self,
        aggregation_type: str,
        dimension: str,
        dimension_value: str,
        period_start: datetime,
        period_end: datetime,
    ) -> FeedbackAggregation:
        """获取或创建聚合数据"""
        try:
            result = await self.db.execute(
                select(FeedbackAggregation).where(
                    FeedbackAggregation.aggregation_type == aggregation_type,
                    FeedbackAggregation.dimension == dimension,
                    FeedbackAggregation.dimension_value == dimension_value,
                    FeedbackAggregation.period_start == period_start,
                )
            )
            aggregation = result.scalar_one_or_none()

            if not aggregation:
                aggregation = FeedbackAggregation(
                    aggregation_type=aggregation_type,
                    dimension=dimension,
                    dimension_value=dimension_value,
                    period_start=period_start,
                    period_end=period_end,
                )
                self.db.add(aggregation)
                await self.db.commit()
                await self.db.refresh(aggregation)

            return aggregation
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"获取或创建聚合数据失败: {e}")
            raise

    async def get_feedback_trends(self, days: int = 30, feedback_type: str = None) -> List[Dict]:
        """获取反馈趋势数据"""
        try:
            end_date = utc_now()
            start_date = end_date - timedelta(days=days)

            date_bucket = func.date_trunc('day', FeedbackEvent.timestamp).label('date')
            stmt = select(
                date_bucket,
                func.count(FeedbackEvent.id).label('count'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality'),
            ).where(
                FeedbackEvent.timestamp >= start_date,
                FeedbackEvent.timestamp <= end_date,
                FeedbackEvent.valid.is_(True),
            )

            if feedback_type:
                stmt = stmt.where(FeedbackEvent.feedback_type == feedback_type)

            stmt = stmt.group_by(date_bucket).order_by(date_bucket)
            results = (await self.db.execute(stmt)).all()

            return [
                {
                    'date': r.date,
                    'count': r.count,
                    'average_quality': float(r.avg_quality) if r.avg_quality else 0.0,
                }
                for r in results
            ]
        except SQLAlchemyError as e:
            logger.error(f"获取反馈趋势失败: {e}")
            raise

    async def get_top_items_by_feedback(self, feedback_type: str = None, limit: int = 10) -> List[Dict]:
        """获取反馈最多的推荐项"""
        try:
            feedback_count = func.count(FeedbackEvent.id).label('feedback_count')
            unique_users = func.count(func.distinct(FeedbackEvent.user_id)).label('unique_users')
            avg_quality = func.avg(FeedbackEvent.quality_score).label('avg_quality')

            stmt = select(
                FeedbackEvent.item_id,
                feedback_count,
                unique_users,
                avg_quality,
            ).where(FeedbackEvent.valid.is_(True))

            if feedback_type:
                stmt = stmt.where(FeedbackEvent.feedback_type == feedback_type)

            stmt = stmt.group_by(FeedbackEvent.item_id).order_by(desc(feedback_count)).limit(limit)
            results = (await self.db.execute(stmt)).all()

            return [
                {
                    'item_id': r.item_id,
                    'feedback_count': r.feedback_count,
                    'unique_users': r.unique_users,
                    'average_quality': float(r.avg_quality) if r.avg_quality else 0.0,
                }
                for r in results
            ]
        except SQLAlchemyError as e:
            logger.error(f"获取热门推荐项失败: {e}")
            raise

    async def get_system_health_metrics(self) -> Dict:
        """获取系统健康指标"""
        try:
            total_events = (await self.db.execute(select(func.count(FeedbackEvent.id)))).scalar() or 0
            processed_events = (await self.db.execute(
                select(func.count(FeedbackEvent.id)).where(FeedbackEvent.processed.is_(True))
            )).scalar() or 0

            avg_quality = (await self.db.execute(
                select(func.avg(FeedbackEvent.quality_score)).where(FeedbackEvent.valid.is_(True))
            )).scalar() or 0

            last_24h = utc_now() - timedelta(hours=24)
            recent_events = (await self.db.execute(
                select(func.count(FeedbackEvent.id)).where(FeedbackEvent.timestamp >= last_24h)
            )).scalar() or 0

            avg_processing_delay = (await self.db.execute(
                select(
                    func.avg(
                        func.extract('epoch', FeedbackEvent.processed_at - FeedbackEvent.timestamp)
                    )
                ).where(
                    FeedbackEvent.processed.is_(True),
                    FeedbackEvent.processed_at.isnot(None),
                )
            )).scalar() or 0

            return {
                'total_events': total_events,
                'processed_events': processed_events,
                'processing_rate': (processed_events / total_events) if total_events > 0 else 0.0,
                'average_quality_score': float(avg_quality) if avg_quality else 0.0,
                'events_last_24h': recent_events,
                'average_processing_delay_seconds': float(avg_processing_delay) if avg_processing_delay else 0.0,
            }
        except SQLAlchemyError as e:
            logger.error(f"获取系统健康指标失败: {e}")
            raise
