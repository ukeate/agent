"""
反馈数据访问层

提供反馈数据的CRUD操作和查询功能
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import SQLAlchemyError

from ..models.schemas.feedback import (
    FeedbackEvent, FeedbackBatch, UserFeedbackProfile, 
    ItemFeedbackSummary, RewardSignal, FeedbackQualityLog,
    FeedbackAggregation
)
from ..core.database import get_db_session

logger = logging.getLogger(__name__)

class FeedbackRepository:
    """反馈数据仓库"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db_session())
    
    # === 反馈事件操作 ===
    
    async def create_feedback_event(self, event_data: Dict) -> FeedbackEvent:
        """创建反馈事件"""
        try:
            event = FeedbackEvent(**event_data)
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            return event
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"创建反馈事件失败: {e}")
            raise
    
    async def create_feedback_batch(self, events: List[Dict], batch_data: Dict) -> FeedbackBatch:
        """批量创建反馈事件"""
        try:
            # 创建批次记录
            batch = FeedbackBatch(**batch_data)
            self.db.add(batch)
            self.db.flush()  # 获取批次ID
            
            # 创建事件记录
            event_objects = []
            for event_data in events:
                event_data['batch_id'] = batch.id
                event = FeedbackEvent(**event_data)
                event_objects.append(event)
            
            self.db.add_all(event_objects)
            
            # 更新批次统计
            batch.event_count = len(event_objects)
            
            self.db.commit()
            self.db.refresh(batch)
            return batch
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"批量创建反馈事件失败: {e}")
            raise
    
    async def get_feedback_events(self, 
                                  user_id: str = None,
                                  item_id: str = None,
                                  feedback_types: List[str] = None,
                                  start_date: datetime = None,
                                  end_date: datetime = None,
                                  limit: int = 100,
                                  offset: int = 0) -> List[FeedbackEvent]:
        """查询反馈事件"""
        try:
            query = self.db.query(FeedbackEvent).filter(FeedbackEvent.valid == True)
            
            if user_id:
                query = query.filter(FeedbackEvent.user_id == user_id)
            
            if item_id:
                query = query.filter(FeedbackEvent.item_id == item_id)
            
            if feedback_types:
                query = query.filter(FeedbackEvent.feedback_type.in_(feedback_types))
            
            if start_date:
                query = query.filter(FeedbackEvent.timestamp >= start_date)
            
            if end_date:
                query = query.filter(FeedbackEvent.timestamp <= end_date)
            
            query = query.order_by(desc(FeedbackEvent.timestamp))
            query = query.offset(offset).limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"查询反馈事件失败: {e}")
            raise
    
    async def update_feedback_processing_status(self, event_ids: List[str], processed: bool = True):
        """更新反馈处理状态"""
        try:
            self.db.query(FeedbackEvent).filter(
                FeedbackEvent.event_id.in_(event_ids)
            ).update({
                'processed': processed,
                'processed_at': utc_now() if processed else None
            }, synchronize_session=False)
            
            self.db.commit()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"更新反馈处理状态失败: {e}")
            raise
    
    # === 用户档案操作 ===
    
    async def get_or_create_user_profile(self, user_id: str) -> UserFeedbackProfile:
        """获取或创建用户反馈档案"""
        try:
            profile = self.db.query(UserFeedbackProfile).filter(
                UserFeedbackProfile.user_id == user_id
            ).first()
            
            if not profile:
                profile = UserFeedbackProfile(user_id=user_id)
                self.db.add(profile)
                self.db.commit()
                self.db.refresh(profile)
            
            return profile
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"获取或创建用户档案失败: {e}")
            raise
    
    async def update_user_profile(self, user_id: str, profile_data: Dict):
        """更新用户反馈档案"""
        try:
            self.db.query(UserFeedbackProfile).filter(
                UserFeedbackProfile.user_id == user_id
            ).update(profile_data)
            
            self.db.commit()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"更新用户档案失败: {e}")
            raise
    
    async def get_user_feedback_stats(self, user_id: str) -> Dict:
        """获取用户反馈统计"""
        try:
            # 基础统计
            stats = self.db.query(
                func.count(FeedbackEvent.id).label('total_count'),
                func.count(func.distinct(FeedbackEvent.item_id)).label('unique_items'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality'),
                func.min(FeedbackEvent.timestamp).label('first_feedback'),
                func.max(FeedbackEvent.timestamp).label('last_feedback')
            ).filter(
                FeedbackEvent.user_id == user_id,
                FeedbackEvent.valid == True
            ).first()
            
            # 类型分布
            type_distribution = self.db.query(
                FeedbackEvent.feedback_type,
                func.count(FeedbackEvent.id).label('count')
            ).filter(
                FeedbackEvent.user_id == user_id,
                FeedbackEvent.valid == True
            ).group_by(FeedbackEvent.feedback_type).all()
            
            return {
                'total_feedbacks': stats.total_count or 0,
                'unique_items': stats.unique_items or 0,
                'average_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
                'first_feedback_time': stats.first_feedback,
                'last_feedback_time': stats.last_feedback,
                'type_distribution': {t.feedback_type: t.count for t in type_distribution}
            }
            
        except SQLAlchemyError as e:
            logger.error(f"获取用户反馈统计失败: {e}")
            raise
    
    # === 推荐项汇总操作 ===
    
    async def get_or_create_item_summary(self, item_id: str) -> ItemFeedbackSummary:
        """获取或创建推荐项汇总"""
        try:
            summary = self.db.query(ItemFeedbackSummary).filter(
                ItemFeedbackSummary.item_id == item_id
            ).first()
            
            if not summary:
                summary = ItemFeedbackSummary(item_id=item_id)
                self.db.add(summary)
                self.db.commit()
                self.db.refresh(summary)
            
            return summary
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"获取或创建推荐项汇总失败: {e}")
            raise
    
    async def update_item_summary(self, item_id: str, summary_data: Dict):
        """更新推荐项汇总"""
        try:
            self.db.query(ItemFeedbackSummary).filter(
                ItemFeedbackSummary.item_id == item_id
            ).update(summary_data)
            
            self.db.commit()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"更新推荐项汇总失败: {e}")
            raise
    
    async def get_item_feedback_stats(self, item_id: str) -> Dict:
        """获取推荐项反馈统计"""
        try:
            # 基础统计
            stats = self.db.query(
                func.count(FeedbackEvent.id).label('total_count'),
                func.count(func.distinct(FeedbackEvent.user_id)).label('unique_users'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality'),
                func.min(FeedbackEvent.timestamp).label('first_feedback'),
                func.max(FeedbackEvent.timestamp).label('last_feedback')
            ).filter(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.valid == True
            ).first()
            
            # 评分统计
            rating_stats = self.db.query(
                func.avg(func.cast(FeedbackEvent.value, 'float')).label('avg_rating'),
                func.count(FeedbackEvent.id).label('rating_count')
            ).filter(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.feedback_type == 'rating',
                FeedbackEvent.valid == True
            ).first()
            
            # 点赞统计
            like_stats = self.db.query(
                func.count(func.case([(FeedbackEvent.feedback_type == 'like', 1)])).label('like_count'),
                func.count(func.case([(FeedbackEvent.feedback_type == 'dislike', 1)])).label('dislike_count')
            ).filter(
                FeedbackEvent.item_id == item_id,
                FeedbackEvent.feedback_type.in_(['like', 'dislike']),
                FeedbackEvent.valid == True
            ).first()
            
            total_likes = like_stats.like_count + like_stats.dislike_count
            like_ratio = like_stats.like_count / total_likes if total_likes > 0 else 0.0
            
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
                'like_ratio': like_ratio
            }
            
        except SQLAlchemyError as e:
            logger.error(f"获取推荐项反馈统计失败: {e}")
            raise
    
    # === 奖励信号操作 ===
    
    async def save_reward_signal(self, reward_data: Dict) -> RewardSignal:
        """保存奖励信号"""
        try:
            # 检查是否已存在
            existing = self.db.query(RewardSignal).filter(
                RewardSignal.user_id == reward_data['user_id'],
                RewardSignal.item_id == reward_data['item_id']
            ).order_by(desc(RewardSignal.calculated_at)).first()
            
            # 如果最近计算的信号还有效，更新它；否则创建新的
            if existing and existing.valid_until and existing.valid_until > utc_now():
                for key, value in reward_data.items():
                    setattr(existing, key, value)
                reward = existing
            else:
                reward = RewardSignal(**reward_data)
                self.db.add(reward)
            
            self.db.commit()
            self.db.refresh(reward)
            return reward
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"保存奖励信号失败: {e}")
            raise
    
    async def get_latest_reward_signal(self, user_id: str, item_id: str) -> Optional[RewardSignal]:
        """获取最新的奖励信号"""
        try:
            return self.db.query(RewardSignal).filter(
                RewardSignal.user_id == user_id,
                RewardSignal.item_id == item_id,
                or_(
                    RewardSignal.valid_until.is_(None),
                    RewardSignal.valid_until > utc_now()
                )
            ).order_by(desc(RewardSignal.calculated_at)).first()
            
        except SQLAlchemyError as e:
            logger.error(f"获取最新奖励信号失败: {e}")
            raise
    
    # === 质量评估操作 ===
    
    async def save_quality_assessment(self, quality_data: Dict) -> FeedbackQualityLog:
        """保存质量评估结果"""
        try:
            quality_log = FeedbackQualityLog(**quality_data)
            self.db.add(quality_log)
            self.db.commit()
            self.db.refresh(quality_log)
            return quality_log
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"保存质量评估失败: {e}")
            raise
    
    # === 聚合数据操作 ===
    
    async def get_or_create_aggregation(self, 
                                        aggregation_type: str,
                                        dimension: str,
                                        dimension_value: str,
                                        period_start: datetime,
                                        period_end: datetime) -> FeedbackAggregation:
        """获取或创建聚合数据"""
        try:
            aggregation = self.db.query(FeedbackAggregation).filter(
                FeedbackAggregation.aggregation_type == aggregation_type,
                FeedbackAggregation.dimension == dimension,
                FeedbackAggregation.dimension_value == dimension_value,
                FeedbackAggregation.period_start == period_start
            ).first()
            
            if not aggregation:
                aggregation = FeedbackAggregation(
                    aggregation_type=aggregation_type,
                    dimension=dimension,
                    dimension_value=dimension_value,
                    period_start=period_start,
                    period_end=period_end
                )
                self.db.add(aggregation)
                self.db.commit()
                self.db.refresh(aggregation)
            
            return aggregation
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"获取或创建聚合数据失败: {e}")
            raise
    
    # === 分析查询 ===
    
    async def get_feedback_trends(self, 
                                  days: int = 30,
                                  feedback_type: str = None) -> List[Dict]:
        """获取反馈趋势数据"""
        try:
            end_date = utc_now()
            start_date = end_date - timedelta(days=days)
            
            query = self.db.query(
                func.date_trunc('day', FeedbackEvent.timestamp).label('date'),
                func.count(FeedbackEvent.id).label('count'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality')
            ).filter(
                FeedbackEvent.timestamp >= start_date,
                FeedbackEvent.timestamp <= end_date,
                FeedbackEvent.valid == True
            )
            
            if feedback_type:
                query = query.filter(FeedbackEvent.feedback_type == feedback_type)
            
            results = query.group_by(
                func.date_trunc('day', FeedbackEvent.timestamp)
            ).order_by('date').all()
            
            return [
                {
                    'date': r.date,
                    'count': r.count,
                    'average_quality': float(r.avg_quality) if r.avg_quality else 0.0
                }
                for r in results
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"获取反馈趋势失败: {e}")
            raise
    
    async def get_top_items_by_feedback(self, 
                                        feedback_type: str = None,
                                        limit: int = 10) -> List[Dict]:
        """获取反馈最多的推荐项"""
        try:
            query = self.db.query(
                FeedbackEvent.item_id,
                func.count(FeedbackEvent.id).label('feedback_count'),
                func.count(func.distinct(FeedbackEvent.user_id)).label('unique_users'),
                func.avg(FeedbackEvent.quality_score).label('avg_quality')
            ).filter(FeedbackEvent.valid == True)
            
            if feedback_type:
                query = query.filter(FeedbackEvent.feedback_type == feedback_type)
            
            results = query.group_by(FeedbackEvent.item_id).order_by(
                desc('feedback_count')
            ).limit(limit).all()
            
            return [
                {
                    'item_id': r.item_id,
                    'feedback_count': r.feedback_count,
                    'unique_users': r.unique_users,
                    'average_quality': float(r.avg_quality) if r.avg_quality else 0.0
                }
                for r in results
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"获取热门推荐项失败: {e}")
            raise
    
    async def get_system_health_metrics(self) -> Dict:
        """获取系统健康指标"""
        try:
            # 基础指标
            total_events = self.db.query(func.count(FeedbackEvent.id)).scalar()
            processed_events = self.db.query(func.count(FeedbackEvent.id)).filter(
                FeedbackEvent.processed == True
            ).scalar()
            
            # 质量指标
            avg_quality = self.db.query(
                func.avg(FeedbackEvent.quality_score)
            ).filter(FeedbackEvent.valid == True).scalar()
            
            # 最近24小时的活动
            last_24h = utc_now() - timedelta(hours=24)
            recent_events = self.db.query(func.count(FeedbackEvent.id)).filter(
                FeedbackEvent.timestamp >= last_24h
            ).scalar()
            
            # 处理延迟
            avg_processing_delay = self.db.query(
                func.avg(
                    func.extract('epoch', FeedbackEvent.processed_at - FeedbackEvent.timestamp)
                )
            ).filter(
                FeedbackEvent.processed == True,
                FeedbackEvent.processed_at.isnot(None)
            ).scalar()
            
            return {
                'total_events': total_events or 0,
                'processed_events': processed_events or 0,
                'processing_rate': (processed_events / total_events) if total_events > 0 else 0.0,
                'average_quality_score': float(avg_quality) if avg_quality else 0.0,
                'events_last_24h': recent_events or 0,
                'average_processing_delay_seconds': float(avg_processing_delay) if avg_processing_delay else 0.0
            }
            
        except SQLAlchemyError as e:
            logger.error(f"获取系统健康指标失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()

# 全局仓库实例
feedback_repository = FeedbackRepository()