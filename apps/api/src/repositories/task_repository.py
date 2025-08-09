"""
任务仓储实现
提供任务相关的数据访问操作
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import logging

from .base import BaseRepository
from ..models.database.workflow import Task

logger = logging.getLogger(__name__)


class TaskRepository(BaseRepository[Task, str]):
    """任务仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Task)
    
    async def get_by_agent_id(self, agent_id: str, limit: int = 50) -> List[Task]:
        """根据智能体ID获取任务列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.agent_id == agent_id)
                .order_by(desc(self.model_class.created_at))
                .limit(limit)
            )
            tasks = result.scalars().all()
            logger.debug(f"根据智能体ID {agent_id} 获取任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"根据智能体ID {agent_id} 获取任务失败: {str(e)}")
            raise
    
    async def get_by_dag_execution_id(self, dag_execution_id: str) -> List[Task]:
        """根据DAG执行ID获取任务列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.dag_execution_id == dag_execution_id)
                .order_by(asc(self.model_class.created_at))
            )
            tasks = result.scalars().all()
            logger.debug(f"根据DAG执行ID {dag_execution_id} 获取任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"根据DAG执行ID {dag_execution_id} 获取任务失败: {str(e)}")
            raise
    
    async def get_by_status(self, status: str, limit: int = 100) -> List[Task]:
        """根据状态获取任务列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.status == status)
                .order_by(desc(self.model_class.updated_at))
                .limit(limit)
            )
            tasks = result.scalars().all()
            logger.debug(f"根据状态 '{status}' 获取任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"根据状态 '{status}' 获取任务失败: {str(e)}")
            raise
    
    async def get_by_type(self, task_type: str, limit: int = 50) -> List[Task]:
        """根据任务类型获取任务列表"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.type == task_type)
                .order_by(desc(self.model_class.created_at))
                .limit(limit)
            )
            tasks = result.scalars().all()
            logger.debug(f"根据任务类型 '{task_type}' 获取任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"根据任务类型 '{task_type}' 获取任务失败: {str(e)}")
            raise
    
    async def get_pending_tasks(self, limit: int = 50) -> List[Task]:
        """获取待处理任务"""
        try:
            result = await self.session.execute(
                select(self.model_class)
                .where(self.model_class.status == 'pending')
                .order_by(asc(self.model_class.created_at))
                .limit(limit)
            )
            tasks = result.scalars().all()
            logger.debug(f"获取待处理任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"获取待处理任务失败: {str(e)}")
            raise
    
    async def get_running_tasks(self, agent_id: Optional[str] = None) -> List[Task]:
        """获取正在运行的任务"""
        try:
            conditions = [self.model_class.status == 'running']
            
            if agent_id:
                conditions.append(self.model_class.agent_id == agent_id)
            
            result = await self.session.execute(
                select(self.model_class)
                .where(and_(*conditions))
                .order_by(desc(self.model_class.updated_at))
            )
            tasks = result.scalars().all()
            logger.debug(f"获取正在运行的任务: {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"获取正在运行的任务失败: {str(e)}")
            raise
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: str, 
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """更新任务状态"""
        try:
            updates = {
                'status': status,
                'updated_at': datetime.utcnow()
            }
            
            if output_data is not None:
                updates['output_data'] = output_data
            
            if error_message:
                updates['error_message'] = error_message
            
            updated_task = await self.update(task_id, updates)
            success = updated_task is not None
            
            if success:
                logger.info(f"更新任务 {task_id} 状态为 '{status}': 成功")
            else:
                logger.warning(f"更新任务 {task_id} 状态失败: 任务不存在")
            
            return success
        except Exception as e:
            logger.error(f"更新任务 {task_id} 状态失败: {str(e)}")
            raise
    
    async def get_overdue_tasks(self, hours: int = 24) -> List[Task]:
        """获取超时任务"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            result = await self.session.execute(
                select(self.model_class)
                .where(
                    and_(
                        self.model_class.status.in_(['pending', 'running']),
                        self.model_class.created_at < cutoff_time
                    )
                )
                .order_by(asc(self.model_class.created_at))
            )
            tasks = result.scalars().all()
            logger.debug(f"获取超时任务 ({hours}小时): {len(tasks)} 个")
            return list(tasks)
        except Exception as e:
            logger.error(f"获取超时任务失败: {str(e)}")
            raise
    
    async def get_task_statistics(
        self,
        agent_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            conditions = []
            
            if agent_id:
                conditions.append(self.model_class.agent_id == agent_id)
            
            if start_date:
                conditions.append(self.model_class.created_at >= start_date)
            
            if end_date:
                conditions.append(self.model_class.created_at <= end_date)
            
            query = select(self.model_class)
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            tasks = result.scalars().all()
            
            # 计算统计信息
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.status == 'completed'])
            failed_tasks = len([t for t in tasks if t.status == 'failed'])
            pending_tasks = len([t for t in tasks if t.status == 'pending'])
            running_tasks = len([t for t in tasks if t.status == 'running'])
            skipped_tasks = len([t for t in tasks if t.status == 'skipped'])
            
            # 按类型统计
            type_stats = {}
            for task in tasks:
                task_type = task.type
                if task_type not in type_stats:
                    type_stats[task_type] = 0
                type_stats[task_type] += 1
            
            statistics = {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'pending_tasks': pending_tasks,
                'running_tasks': running_tasks,
                'skipped_tasks': skipped_tasks,
                'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
                'failure_rate': failed_tasks / total_tasks if total_tasks > 0 else 0,
                'type_distribution': type_stats
            }
            
            logger.debug(f"获取任务统计信息: {statistics}")
            return statistics
        except Exception as e:
            logger.error(f"获取任务统计信息失败: {str(e)}")
            raise
    
    async def search_tasks(
        self,
        query: Optional[str] = None,
        agent_id: Optional[str] = None,
        dag_execution_id: Optional[str] = None,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Task]:
        """搜索任务"""
        try:
            conditions = []
            
            # 文本搜索（基于任务名称）
            if query:
                conditions.append(
                    self.model_class.name.ilike(f'%{query}%')
                )
            
            # 过滤条件
            if agent_id:
                conditions.append(self.model_class.agent_id == agent_id)
            
            if dag_execution_id:
                conditions.append(self.model_class.dag_execution_id == dag_execution_id)
            
            if status:
                conditions.append(self.model_class.status == status)
            
            if task_type:
                conditions.append(self.model_class.type == task_type)
            
            query_stmt = select(self.model_class)
            if conditions:
                query_stmt = query_stmt.where(and_(*conditions))
            
            query_stmt = query_stmt.order_by(desc(self.model_class.updated_at)).limit(limit)
            
            result = await self.session.execute(query_stmt)
            tasks = result.scalars().all()
            
            logger.debug(f"搜索任务: {len(tasks)} 个结果")
            return list(tasks)
        except Exception as e:
            logger.error(f"搜索任务失败: {str(e)}")
            raise
    
    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """分配任务给智能体"""
        try:
            updated_task = await self.update(task_id, {
                'agent_id': agent_id,
                'status': 'pending',
                'updated_at': datetime.utcnow()
            })
            
            success = updated_task is not None
            if success:
                logger.info(f"分配任务 {task_id} 给智能体 {agent_id}: 成功")
            else:
                logger.warning(f"分配任务 {task_id} 失败: 任务不存在")
            
            return success
        except Exception as e:
            logger.error(f"分配任务 {task_id} 给智能体 {agent_id} 失败: {str(e)}")
            raise