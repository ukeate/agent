"""
Supervisor数据访问层实现
提供Supervisor相关数据库操作
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, and_, or_, func
import structlog

from ..repositories.base import BaseRepository
from ..models.database.supervisor import (
    SupervisorAgent, SupervisorTask, SupervisorDecision, 
    AgentLoadMetrics, SupervisorConfig
)
from ..models.schemas.supervisor import TaskStatus, TaskType, TaskPriority, AgentStatus

logger = structlog.get_logger(__name__)


class SupervisorRepository(BaseRepository[SupervisorAgent, str]):
    """Supervisor智能体仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SupervisorAgent)
    
    async def get_by_name(self, name: str) -> Optional[SupervisorAgent]:
        """根据名称获取Supervisor"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorAgent).where(SupervisorAgent.name == name)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("获取Supervisor失败", name=name, error=str(e))
            return None
    
    async def get_active_supervisors(self) -> List[SupervisorAgent]:
        """获取所有活跃的Supervisor"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorAgent).where(SupervisorAgent.status == AgentStatus.ACTIVE.value)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("获取活跃Supervisor失败", error=str(e))
            return []
    
    async def update_status(self, supervisor_id: str, status: AgentStatus) -> bool:
        """更新Supervisor状态"""
        try:
            from sqlalchemy import update
            stmt = update(SupervisorAgent).where(
                SupervisorAgent.id == supervisor_id
            ).values(
                status=status.value,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            if result.rowcount > 0:
                logger.info("Supervisor状态已更新", supervisor_id=supervisor_id, status=status.value)
                return True
            return False
        except Exception as e:
            logger.error("更新Supervisor状态失败", supervisor_id=supervisor_id, error=str(e))
            await self.session.rollback()
            return False
    
    async def update_performance_metrics(
        self, 
        supervisor_id: str, 
        metrics: Dict[str, Any]
    ) -> bool:
        """更新Supervisor性能指标"""
        try:
            from sqlalchemy import update
            stmt = update(SupervisorAgent).where(
                SupervisorAgent.id == supervisor_id
            ).values(
                performance_metrics=metrics,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            if result.rowcount > 0:
                return True
            return False
        except Exception as e:
            logger.error("更新性能指标失败", supervisor_id=supervisor_id, error=str(e))
            await self.session.rollback()
            return False


class SupervisorTaskRepository(BaseRepository[SupervisorTask, str]):
    """Supervisor任务仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SupervisorTask)
    
    async def get_by_supervisor_id(
        self, 
        supervisor_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[SupervisorTask]:
        """根据Supervisor ID获取任务列表"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorTask).where(
                SupervisorTask.supervisor_id == supervisor_id
            ).order_by(desc(SupervisorTask.created_at)).limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("获取Supervisor任务失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def get_tasks_by_supervisor(
        self,
        supervisor_id: str,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[SupervisorTask]:
        """根据Supervisor ID或名称获取任务列表"""
        try:
            from sqlalchemy import select
            
            # 首先尝试通过supervisor_id查询
            conditions = []
            
            # 尝试按ID查询
            conditions.append(SupervisorTask.supervisor_id == supervisor_id)
            
            # 如果没有找到，尝试通过supervisor名称查询
            # 这需要先查询supervisor表获取对应的ID
            supervisor_repo = SupervisorRepository(self.session)
            supervisor = await supervisor_repo.get_by_name(supervisor_id)
            if supervisor:
                conditions.append(SupervisorTask.supervisor_id == supervisor.id)
            
            # 构建查询条件
            where_condition = or_(*conditions) if len(conditions) > 1 else conditions[0]
            
            stmt = select(SupervisorTask).where(where_condition)
            
            # 添加状态过滤
            if status_filter:
                stmt = stmt.where(SupervisorTask.status == status_filter)
            
            stmt = stmt.order_by(desc(SupervisorTask.created_at)).limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error("获取任务列表失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def get_pending_tasks(self, supervisor_id: str) -> List[SupervisorTask]:
        """获取待处理任务"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorTask).where(
                and_(
                    SupervisorTask.supervisor_id == supervisor_id,
                    SupervisorTask.status == TaskStatus.PENDING.value
                )
            ).order_by(SupervisorTask.priority, SupervisorTask.created_at)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("获取待处理任务失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def get_running_tasks(self, supervisor_id: str) -> List[SupervisorTask]:
        """获取正在运行的任务"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorTask).where(
                and_(
                    SupervisorTask.supervisor_id == supervisor_id,
                    SupervisorTask.status == TaskStatus.RUNNING.value
                )
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("获取运行中任务失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def get_tasks_by_status(
        self, 
        supervisor_id: str, 
        status: TaskStatus
    ) -> List[SupervisorTask]:
        """根据状态获取任务"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorTask).where(
                and_(
                    SupervisorTask.supervisor_id == supervisor_id,
                    SupervisorTask.status == status.value
                )
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("按状态获取任务失败", supervisor_id=supervisor_id, status=status.value, error=str(e))
            return []
    
    async def get_tasks_by_agent(
        self, 
        supervisor_id: str, 
        agent_name: str,
        limit: int = 20
    ) -> List[SupervisorTask]:
        """根据分配的智能体获取任务"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorTask).where(
                and_(
                    SupervisorTask.supervisor_id == supervisor_id,
                    SupervisorTask.assigned_agent_name == agent_name
                )
            ).order_by(desc(SupervisorTask.created_at)).limit(limit)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("按智能体获取任务失败", agent_name=agent_name, error=str(e))
            return []
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus,
        output_data: Optional[Dict[str, Any]] = None,
        actual_time_seconds: Optional[int] = None
    ) -> bool:
        """更新任务状态"""
        try:
            from sqlalchemy import update
            update_data = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc)
            }
            
            if status == TaskStatus.RUNNING:
                update_data["started_at"] = datetime.now(timezone.utc)
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                update_data["completed_at"] = datetime.now(timezone.utc)
            
            if output_data is not None:
                update_data["output_data"] = output_data
            
            if actual_time_seconds is not None:
                update_data["actual_time_seconds"] = actual_time_seconds
            
            stmt = update(SupervisorTask).where(
                SupervisorTask.id == task_id
            ).values(**update_data)
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            if result.rowcount > 0:
                logger.info("任务状态已更新", task_id=task_id, status=status.value)
                return True
            return False
        except Exception as e:
            logger.error("更新任务状态失败", task_id=task_id, error=str(e))
            await self.session.rollback()
            return False
    
    async def assign_task_to_agent(
        self, 
        task_id: str, 
        agent_id: str, 
        agent_name: str
    ) -> bool:
        """将任务分配给智能体"""
        try:
            from sqlalchemy import update
            stmt = update(SupervisorTask).where(
                SupervisorTask.id == task_id
            ).values(
                assigned_agent_id=agent_id,
                assigned_agent_name=agent_name,
                status=TaskStatus.PENDING.value,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            if result.rowcount > 0:
                logger.info("任务已分配", task_id=task_id, agent_name=agent_name)
                return True
            return False
        except Exception as e:
            logger.error("分配任务失败", task_id=task_id, agent_name=agent_name, error=str(e))
            await self.session.rollback()
            return False
    
    async def get_task_statistics(self, supervisor_id: str) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            from sqlalchemy import select
            # 按状态统计
            status_stmt = select(
                SupervisorTask.status,
                func.count(SupervisorTask.id).label('count')
            ).where(
                SupervisorTask.supervisor_id == supervisor_id
            ).group_by(SupervisorTask.status)
            status_result = await self.session.execute(status_stmt)
            status_counts = status_result.fetchall()
            
            # 按类型统计
            type_stmt = select(
                SupervisorTask.task_type,
                func.count(SupervisorTask.id).label('count')
            ).where(
                SupervisorTask.supervisor_id == supervisor_id
            ).group_by(SupervisorTask.task_type)
            type_result = await self.session.execute(type_stmt)
            type_counts = type_result.fetchall()
            
            # 按优先级统计
            priority_stmt = select(
                SupervisorTask.priority,
                func.count(SupervisorTask.id).label('count')
            ).where(
                SupervisorTask.supervisor_id == supervisor_id
            ).group_by(SupervisorTask.priority)
            priority_result = await self.session.execute(priority_stmt)
            priority_counts = priority_result.fetchall()
            
            return {
                "status_distribution": {status: count for status, count in status_counts},
                "type_distribution": {task_type: count for task_type, count in type_counts},
                "priority_distribution": {priority: count for priority, count in priority_counts}
            }
        except Exception as e:
            logger.error("获取任务统计失败", supervisor_id=supervisor_id, error=str(e))
            return {}


class SupervisorDecisionRepository(BaseRepository[SupervisorDecision, str]):
    """Supervisor决策记录仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SupervisorDecision)
    
    async def get_by_supervisor_id(
        self, 
        supervisor_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[SupervisorDecision]:
        """根据Supervisor ID或名称获取决策历史"""
        try:
            from sqlalchemy import select
            
            # 首先尝试直接按ID查询
            stmt = select(SupervisorDecision).where(
                SupervisorDecision.supervisor_id == supervisor_id
            ).order_by(desc(SupervisorDecision.timestamp)).limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            decisions = list(result.scalars().all())
            
            # 如果按ID没找到，尝试通过supervisor名称查询
            if not decisions:
                supervisor_repo = SupervisorRepository(self.session)
                supervisor = await supervisor_repo.get_by_name(supervisor_id)
                if supervisor:
                    stmt = select(SupervisorDecision).where(
                        SupervisorDecision.supervisor_id == supervisor.id
                    ).order_by(desc(SupervisorDecision.timestamp)).limit(limit).offset(offset)
                    result = await self.session.execute(stmt)
                    decisions = list(result.scalars().all())
            
            return decisions
        except Exception as e:
            logger.error("获取决策历史失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def get_by_task_id(self, task_id: str) -> Optional[SupervisorDecision]:
        """根据任务ID获取决策记录"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorDecision).where(
                SupervisorDecision.task_id == task_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("获取任务决策记录失败", task_id=task_id, error=str(e))
            return None
    
    async def get_successful_decisions(
        self, 
        supervisor_id: str, 
        days: int = 30
    ) -> List[SupervisorDecision]:
        """获取成功的决策记录（用于学习）"""
        try:
            from sqlalchemy import select
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = select(SupervisorDecision).where(
                and_(
                    SupervisorDecision.supervisor_id == supervisor_id,
                    SupervisorDecision.task_success == True,
                    SupervisorDecision.timestamp >= cutoff_date
                )
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("获取成功决策记录失败", supervisor_id=supervisor_id, error=str(e))
            return []
    
    async def update_decision_outcome(
        self, 
        decision_id: str, 
        task_success: bool,
        actual_completion_time: datetime,
        quality_score: Optional[float] = None
    ) -> bool:
        """更新决策结果"""
        try:
            from sqlalchemy import update
            update_data = {
                "task_success": task_success,
                "actual_completion_time": actual_completion_time
            }
            
            if quality_score is not None:
                update_data["quality_score"] = quality_score
            
            stmt = update(SupervisorDecision).where(
                SupervisorDecision.decision_id == decision_id
            ).values(**update_data)
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            if result.rowcount > 0:
                logger.info("决策结果已更新", decision_id=decision_id, task_success=task_success)
                return True
            return False
        except Exception as e:
            logger.error("更新决策结果失败", decision_id=decision_id, error=str(e))
            await self.session.rollback()
            return False
    
    async def get_decision_statistics(self, supervisor_id: str) -> Dict[str, Any]:
        """获取决策统计信息"""
        try:
            from sqlalchemy import select
            # 总决策数
            total_stmt = select(func.count(SupervisorDecision.id)).where(
                SupervisorDecision.supervisor_id == supervisor_id
            )
            total_result = await self.session.execute(total_stmt)
            total_decisions = total_result.scalar()
            
            # 成功决策数
            success_stmt = select(func.count(SupervisorDecision.id)).where(
                and_(
                    SupervisorDecision.supervisor_id == supervisor_id,
                    SupervisorDecision.task_success == True
                )
            )
            success_result = await self.session.execute(success_stmt)
            successful_decisions = success_result.scalar()
            
            # 平均置信度
            confidence_stmt = select(
                func.avg(SupervisorDecision.confidence_level)
            ).where(
                SupervisorDecision.supervisor_id == supervisor_id
            )
            confidence_result = await self.session.execute(confidence_stmt)
            avg_confidence = confidence_result.scalar() or 0.0
            
            # 平均质量分数
            quality_stmt = select(
                func.avg(SupervisorDecision.quality_score)
            ).where(
                and_(
                    SupervisorDecision.supervisor_id == supervisor_id,
                    SupervisorDecision.quality_score.isnot(None)
                )
            )
            quality_result = await self.session.execute(quality_stmt)
            avg_quality = quality_result.scalar() or 0.0
            
            success_rate = (successful_decisions / total_decisions) if total_decisions > 0 else 0.0
            
            return {
                "total_decisions": total_decisions,
                "successful_decisions": successful_decisions,
                "success_rate": success_rate,
                "average_confidence": float(avg_confidence),
                "average_quality_score": float(avg_quality)
            }
        except Exception as e:
            logger.error("获取决策统计失败", supervisor_id=supervisor_id, error=str(e))
            return {}


class AgentLoadMetricsRepository(BaseRepository[AgentLoadMetrics, str]):
    """智能体负载指标仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AgentLoadMetrics)
    
    async def get_current_loads(self, supervisor_id: str) -> Dict[str, float]:
        """获取当前智能体负载"""
        try:
            from sqlalchemy import select
            # 获取最新的负载指标
            stmt = select(AgentLoadMetrics).where(
                AgentLoadMetrics.supervisor_id == supervisor_id
            ).order_by(desc(AgentLoadMetrics.updated_at))
            result = await self.session.execute(stmt)
            latest_metrics = result.scalars().all()
            
            loads = {}
            agent_seen = set()
            
            for metric in latest_metrics:
                if metric.agent_name not in agent_seen:
                    loads[metric.agent_name] = metric.current_load
                    agent_seen.add(metric.agent_name)
            
            return loads
        except Exception as e:
            logger.error("获取当前负载失败", supervisor_id=supervisor_id, error=str(e))
            return {}
    
    async def update_agent_load(
        self, 
        agent_name: str, 
        supervisor_id: str,
        load_change: float
    ) -> bool:
        """更新智能体负载"""
        try:
            from sqlalchemy import select, update
            # 获取最新负载记录
            stmt = select(AgentLoadMetrics).where(
                and_(
                    AgentLoadMetrics.agent_name == agent_name,
                    AgentLoadMetrics.supervisor_id == supervisor_id
                )
            ).order_by(desc(AgentLoadMetrics.updated_at))
            result = await self.session.execute(stmt)
            latest_metric = result.scalars().first()
            
            if latest_metric:
                new_load = max(0.0, min(1.0, latest_metric.current_load + load_change))
                update_stmt = update(AgentLoadMetrics).where(
                    AgentLoadMetrics.id == latest_metric.id
                ).values(
                    current_load=new_load,
                    updated_at=datetime.now(timezone.utc),
                    task_count=latest_metric.task_count + (1 if load_change > 0 else 0)
                )
                await self.session.execute(update_stmt)
            else:
                # 创建新记录
                new_metric = AgentLoadMetrics(
                    agent_name=agent_name,
                    supervisor_id=supervisor_id,
                    current_load=max(0.0, min(1.0, load_change)),
                    task_count=1 if load_change > 0 else 0,
                    window_start=datetime.now(timezone.utc),
                    window_end=datetime.now(timezone.utc) + timedelta(hours=1)
                )
                self.session.add(new_metric)
            
            await self.session.commit()
            return True
        except Exception as e:
            logger.error("更新智能体负载失败", agent_name=agent_name, error=str(e))
            await self.session.rollback()
            return False


class SupervisorConfigRepository(BaseRepository[SupervisorConfig, str]):
    """Supervisor配置仓储"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SupervisorConfig)
    
    async def get_active_config(self, supervisor_id: str) -> Optional[SupervisorConfig]:
        """获取活跃的配置"""
        try:
            from sqlalchemy import select
            stmt = select(SupervisorConfig).where(
                and_(
                    SupervisorConfig.supervisor_id == supervisor_id,
                    SupervisorConfig.is_active == True
                )
            ).order_by(desc(SupervisorConfig.created_at))
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error("获取活跃配置失败", supervisor_id=supervisor_id, error=str(e))
            return None
    
    async def deactivate_all_configs(self, supervisor_id: str) -> bool:
        """停用所有配置"""
        try:
            from sqlalchemy import update
            stmt = update(SupervisorConfig).where(
                SupervisorConfig.supervisor_id == supervisor_id
            ).values(is_active=False)
            
            await self.session.execute(stmt)
            await self.session.commit()
            return True
        except Exception as e:
            logger.error("停用配置失败", supervisor_id=supervisor_id, error=str(e))
            await self.session.rollback()
            return False
    
    async def activate_config(self, config_id: str) -> bool:
        """激活配置"""
        try:
            from sqlalchemy import update
            # 先获取配置
            config = await self.get_by_id(config_id)
            if not config:
                return False
            
            # 停用同一supervisor的所有配置
            await self.deactivate_all_configs(config.supervisor_id)
            
            # 激活指定配置
            stmt = update(SupervisorConfig).where(
                SupervisorConfig.id == config_id
            ).values(is_active=True)
            
            result = await self.session.execute(stmt)
            if result.rowcount > 0:
                await self.session.commit()
                return True
            return False
        except Exception as e:
            logger.error("激活配置失败", config_id=config_id, error=str(e))
            await self.session.rollback()
            return False