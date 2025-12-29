"""
会话仓储实现
提供会话相关的数据访问操作
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from .base import BaseRepository
from src.models.database.session import Session

from src.core.logging import get_logger
logger = get_logger(__name__)

class SessionRepository(BaseRepository[Session, str]):
    """会话仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Session)
    
    async def get_by_user_id(self, user_id: str, limit: int = 20) -> List[Session]:
        """根据用户ID获取会话列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.user_id == user_id)
                .order_by(desc(self.model_class.updated_at))
                .limit(limit)
            )
            sessions = result.scalars().all()
            logger.debug(f"根据用户ID {user_id} 获取会话: {len(sessions)} 个")
            return list(sessions)
        except Exception as e:
            logger.error(f"根据用户ID {user_id} 获取会话失败: {str(e)}")
            raise
    
    async def get_active_sessions(self, limit: int = 50) -> List[Session]:
        """获取活跃会话"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.status == 'active')
                .order_by(desc(self.model_class.updated_at))
                .limit(limit)
            )
            sessions = result.scalars().all()
            logger.debug(f"获取活跃会话: {len(sessions)} 个")
            return list(sessions)
        except Exception as e:
            logger.error(f"获取活跃会话失败: {str(e)}")
            raise
    
    async def get_sessions_by_status(self, status: str, limit: int = 50) -> List[Session]:
        """根据状态获取会话列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.status == status)
                .order_by(desc(self.model_class.updated_at))
                .limit(limit)
            )
            sessions = result.scalars().all()
            logger.debug(f"根据状态 '{status}' 获取会话: {len(sessions)} 个")
            return list(sessions)
        except Exception as e:
            logger.error(f"根据状态 '{status}' 获取会话失败: {str(e)}")
            raise
    
    async def update_message_count(
        self, 
        session_id: str, 
        increment: int = 1
    ) -> bool:
        """更新会话消息计数"""
        try:
            session_obj = await self.get_by_id(session_id)
            if not session_obj:
                logger.warning(f"会话 {session_id} 不存在")
                return False
            
            new_count = session_obj.message_count + increment
            await self.update(session_id, {
                'message_count': new_count,
                'last_activity_at': utc_now(),
                'updated_at': utc_now()
            })
            
            logger.debug(f"更新会话 {session_id} 消息计数: +{increment}")
            return True
        except Exception as e:
            logger.error(f"更新会话 {session_id} 消息计数失败: {str(e)}")
            raise
    
    async def update_context(
        self, 
        session_id: str, 
        context_updates: Dict[str, Any]
    ) -> bool:
        """更新会话上下文"""
        try:
            session_obj = await self.get_by_id(session_id)
            if not session_obj:
                return False
            
            # 合并上下文更新
            current_context = session_obj.context or {}
            current_context.update(context_updates)
            
            await self.update(session_id, {
                'context': current_context,
                'last_activity_at': utc_now(),
                'updated_at': utc_now()
            })
            
            logger.debug(f"更新会话 {session_id} 上下文: {list(context_updates.keys())}")
            return True
        except Exception as e:
            logger.error(f"更新会话 {session_id} 上下文失败: {str(e)}")
            raise
    
    async def update_session_status(self, session_id: str, status: str) -> bool:
        """更新会话状态"""
        try:
            updated_session = await self.update(session_id, {
                'status': status,
                'updated_at': utc_now()
            })
            
            success = updated_session is not None
            if success:
                logger.info(f"更新会话 {session_id} 状态为 '{status}': 成功")
            else:
                logger.warning(f"更新会话 {session_id} 状态失败: 会话不存在")
            
            return success
        except Exception as e:
            logger.error(f"更新会话 {session_id} 状态失败: {str(e)}")
            raise
    
    async def get_expired_sessions(self, expiry_hours: int = 24) -> List[Session]:
        """获取过期会话"""
        try:
            expiry_time = utc_now() - timedelta(hours=expiry_hours)
            
            result = await self.session.execute(
                select(self.model_class)
                .where(
                    and_(
                        self.model_class.updated_at < expiry_time,
                        self.model_class.status == 'active'
                    )
                )
                .order_by(desc(self.model_class.updated_at))
            )
            sessions = result.scalars().all()
            logger.debug(f"获取过期会话 ({expiry_hours}小时): {len(sessions)} 个")
            return list(sessions)
        except Exception as e:
            logger.error(f"获取过期会话失败: {str(e)}")
            raise
    
    async def cleanup_expired_sessions(self, expiry_hours: int = 24) -> int:
        """清理过期会话"""
        try:
            expired_sessions = await self.get_expired_sessions(expiry_hours)
            cleanup_count = 0
            
            for session in expired_sessions:
                await self.update_session_status(session.id, 'expired')
                cleanup_count += 1
            
            logger.info(f"清理过期会话: {cleanup_count} 个")
            return cleanup_count
        except Exception as e:
            logger.error(f"清理过期会话失败: {str(e)}")
            raise
    
    async def get_session_statistics(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取会话统计信息"""
        try:
            query = select(self.model_class)
            
            # 添加时间过滤
            conditions = []
            if start_date:
                conditions.append(self.model_class.created_at >= start_date)
            if end_date:
                conditions.append(self.model_class.created_at <= end_date)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            sessions = result.scalars().all()
            
            # 计算统计信息
            total_sessions = len(sessions)
            active_sessions = len([s for s in sessions if s.status == 'active'])
            completed_sessions = len([s for s in sessions if s.status == 'completed'])
            
            # 计算消息总数
            total_messages = sum(
                session.message_count or 0
                for session in sessions
            )
            
            statistics = {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'completed_sessions': completed_sessions,
                'total_messages': total_messages,
                'average_messages_per_session': total_messages / total_sessions if total_sessions > 0 else 0
            }
            
            logger.debug(f"获取会话统计信息: {statistics}")
            return statistics
        except Exception as e:
            logger.error(f"获取会话统计信息失败: {str(e)}")
            raise
    
    async def search_sessions(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[Session]:
        """搜索会话"""
        try:
            conditions = []
            
            # 添加搜索条件
            if query:
                # 这里可以搜索会话中的消息内容，具体实现依赖于数据库
                conditions.append(self.model_class.title.ilike(f"%{query}%"))
            
            if user_id:
                conditions.append(self.model_class.user_id == user_id)
            
            if status:
                conditions.append(self.model_class.status == status)
            
            query_stmt = select(self.model_class)
            if conditions:
                query_stmt = query_stmt.where(and_(*conditions))
            
            query_stmt = query_stmt.order_by(desc(self.model_class.updated_at)).limit(limit)
            
            result = await self.session.execute(query_stmt)
            sessions = result.scalars().all()
            
            logger.debug(f"搜索会话: {len(sessions)} 个结果")
            return list(sessions)
        except Exception as e:
            logger.error(f"搜索会话失败: {str(e)}")
            raise
