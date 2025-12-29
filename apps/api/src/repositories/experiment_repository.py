"""
A/B测试实验平台数据访问层
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from src.models.database.experiment import (
    Experiment, ExperimentVariant, ExperimentAssignment,
    ExperimentEvent, ExperimentMetricResult
)
from src.models.schemas.experiment import (
    ExperimentConfig, CreateExperimentRequest, UpdateExperimentRequest,
    ExperimentStatus, RecordEventRequest
)
from src.repositories.base import BaseRepository

class ExperimentRepository(BaseRepository[Experiment, ExperimentConfig]):
    """实验数据访问层"""

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, Experiment)

    async def create_experiment(self, experiment_request: CreateExperimentRequest) -> Experiment:
        """创建新实验"""
        try:
            experiment = Experiment(
                name=experiment_request.name,
                description=experiment_request.description,
                hypothesis=experiment_request.hypothesis,
                owner=experiment_request.owner,
                status=ExperimentStatus.DRAFT,
                start_date=experiment_request.start_date,
                end_date=experiment_request.end_date,
                success_metrics=[m for m in experiment_request.success_metrics],
                guardrail_metrics=[m for m in experiment_request.guardrail_metrics],
                minimum_sample_size=experiment_request.minimum_sample_size,
                significance_level=experiment_request.significance_level,
                power=experiment_request.power,
                layers=[layer for layer in experiment_request.layers],
                targeting_rules=[rule.model_dump() for rule in experiment_request.targeting_rules],
                metadata_=experiment_request.metadata or {},
            )

            self.session.add(experiment)
            await self.session.flush()

            for variant in experiment_request.variants:
                traffic_allocation = next(
                    (alloc for alloc in experiment_request.traffic_allocation
                     if alloc.variant_id == variant.variant_id),
                    None,
                )

                if not traffic_allocation:
                    raise ValueError(f"No traffic allocation found for variant {variant.variant_id}")

                variant_record = ExperimentVariant(
                    experiment_id=experiment.id,
                    variant_id=variant.variant_id,
                    name=variant.name,
                    description=variant.description,
                    config=variant.config,
                    is_control=variant.is_control,
                    traffic_percentage=traffic_allocation.percentage,
                )
                self.session.add(variant_record)

            await self.session.commit()
            await self.session.refresh(experiment)
            return experiment
        except IntegrityError as e:
            await self.session.rollback()
            raise ValueError(f"Failed to create experiment: {str(e)}")

    async def get_experiment_with_variants(self, experiment_id: str) -> Optional[Experiment]:
        """获取实验及其变体信息"""
        result = await self.session.execute(
            select(Experiment)
            .options(joinedload(Experiment.variants))
            .where(Experiment.id == experiment_id)
        )
        return result.scalar_one_or_none()

    async def get_experiments_by_owner(
        self,
        owner: str,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """获取用户的实验列表"""
        stmt = select(Experiment).where(Experiment.owner == owner)
        if status:
            stmt = stmt.where(Experiment.status == status)
        stmt = stmt.order_by(desc(Experiment.created_at))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_running_experiments(self, layer: Optional[str] = None) -> List[Experiment]:
        """获取运行中的实验"""
        stmt = select(Experiment).where(Experiment.status == ExperimentStatus.RUNNING)
        if layer:
            stmt = stmt.where(Experiment.layers.op('?')(layer))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> bool:
        """更新实验状态"""
        try:
            result = await self.session.execute(
                update(Experiment)
                .where(Experiment.id == experiment_id)
                .values(status=status, updated_at=utc_now())
            )
            await self.session.commit()
            return (result.rowcount or 0) > 0
        except Exception:
            await self.session.rollback()
            return False

    async def update_experiment(
        self,
        experiment_id: str,
        update_request: UpdateExperimentRequest,
    ) -> Optional[Experiment]:
        """更新实验配置"""
        try:
            experiment = await self.get_by_id(experiment_id)
            if not experiment:
                return None

            if experiment.status != ExperimentStatus.DRAFT:
                allowed_fields = ['end_date', 'targeting_rules', 'metadata']
                update_data = {
                    k: v
                    for k, v in update_request.model_dump(exclude_unset=True).items()
                    if k in allowed_fields and v is not None
                }
            else:
                update_data = {
                    k: v
                    for k, v in update_request.model_dump(exclude_unset=True).items()
                    if v is not None
                }

            if update_data:
                if 'targeting_rules' in update_data:
                    update_data['targeting_rules'] = [
                        rule.model_dump() for rule in update_data['targeting_rules']
                    ]

                for key, value in update_data.items():
                    setattr(experiment, key, value)

                experiment.updated_at = utc_now()
                await self.session.commit()

            return await self.get_experiment_with_variants(experiment_id)
        except Exception:
            await self.session.rollback()
            return None

    async def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验（仅草稿状态）"""
        try:
            experiment = await self.get_by_id(experiment_id)
            if not experiment:
                return False

            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Only draft experiments can be deleted")

            await self.session.delete(experiment)
            await self.session.commit()
            return True
        except Exception:
            await self.session.rollback()
            return False

    async def check_layer_conflicts(self, experiment_id: str, layers: List[str]) -> List[Dict[str, Any]]:
        """检查实验层冲突"""
        conflicts: List[Dict[str, Any]] = []

        for layer in layers:
            stmt = select(Experiment).where(
                and_(
                    Experiment.id != experiment_id,
                    Experiment.status.in_([ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]),
                    Experiment.layers.op('?')(layer),
                )
            )
            result = await self.session.execute(stmt)
            conflicting_experiments = result.scalars().all()

            for conflict_exp in conflicting_experiments:
                conflicts.append({
                    'layer': layer,
                    'conflicting_experiment_id': conflict_exp.id,
                    'conflicting_experiment_name': conflict_exp.name,
                    'conflict_type': 'mutual_exclusion',
                })

        return conflicts

class ExperimentAssignmentRepository(BaseRepository[ExperimentAssignment, None]):
    """实验分配数据访问层"""

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, ExperimentAssignment)

    async def get_user_assignment(self, experiment_id: str, user_id: str) -> Optional[ExperimentAssignment]:
        """获取用户的实验分配"""
        result = await self.session.execute(
            select(ExperimentAssignment).where(
                and_(
                    ExperimentAssignment.experiment_id == experiment_id,
                    ExperimentAssignment.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def create_assignment(
        self,
        experiment_id: str,
        user_id: str,
        variant_id: str,
        context: Dict[str, Any],
        is_eligible: bool = True,
        assignment_reason: str = "traffic_split",
    ) -> ExperimentAssignment:
        """创建用户实验分配"""
        try:
            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant_id,
                context=context,
                is_eligible=is_eligible,
                assignment_reason=assignment_reason,
            )
            self.session.add(assignment)
            await self.session.commit()
            await self.session.refresh(assignment)
            return assignment
        except IntegrityError:
            await self.session.rollback()
            existing = await self.get_user_assignment(experiment_id, user_id)
            if not existing:
                raise
            return existing

    async def get_experiment_assignments(self, experiment_id: str, limit: int = 1000) -> List[ExperimentAssignment]:
        """获取实验的所有分配"""
        result = await self.session.execute(
            select(ExperimentAssignment)
            .where(ExperimentAssignment.experiment_id == experiment_id)
            .order_by(desc(ExperimentAssignment.timestamp))
            .limit(limit)
        )
        return result.scalars().all()

    async def get_variant_assignment_counts(self, experiment_id: str) -> Dict[str, int]:
        """获取各变体的分配数量"""
        result = await self.session.execute(
            select(
                ExperimentAssignment.variant_id,
                func.count(ExperimentAssignment.id).label('count'),
            )
            .where(ExperimentAssignment.experiment_id == experiment_id)
            .group_by(ExperimentAssignment.variant_id)
        )
        return {row.variant_id: row.count for row in result.all()}

class ExperimentEventRepository(BaseRepository[ExperimentEvent, None]):
    """实验事件数据访问层"""

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, ExperimentEvent)

    async def record_event(self, assignment_id: str, event_request: RecordEventRequest) -> ExperimentEvent:
        """记录实验事件"""
        try:
            assignment_result = await self.session.execute(
                select(ExperimentAssignment).where(ExperimentAssignment.id == assignment_id)
            )
            assignment = assignment_result.scalar_one_or_none()
            if not assignment:
                raise ValueError(f"Assignment {assignment_id} not found")

            event = ExperimentEvent(
                experiment_id=assignment.experiment_id,
                assignment_id=assignment_id,
                variant_id=assignment.variant_id,
                user_id=event_request.user_id,
                event_type=event_request.event_type,
                event_value=event_request.event_value,
                metadata_=event_request.metadata,
            )
            self.session.add(event)
            await self.session.commit()
            await self.session.refresh(event)
            return event
        except Exception:
            await self.session.rollback()
            raise

    async def batch_record_events(self, assignment_id: str, events: List[RecordEventRequest]) -> List[ExperimentEvent]:
        """批量记录实验事件"""
        try:
            assignment_result = await self.session.execute(
                select(ExperimentAssignment).where(ExperimentAssignment.id == assignment_id)
            )
            assignment = assignment_result.scalar_one_or_none()
            if not assignment:
                raise ValueError(f"Assignment {assignment_id} not found")

            event_records: List[ExperimentEvent] = []
            for event_request in events:
                event = ExperimentEvent(
                    experiment_id=assignment.experiment_id,
                    assignment_id=assignment_id,
                    variant_id=assignment.variant_id,
                    user_id=event_request.user_id,
                    event_type=event_request.event_type,
                    event_value=event_request.event_value,
                    metadata_=event_request.metadata,
                )
                event_records.append(event)

            self.session.add_all(event_records)
            await self.session.commit()
            return event_records
        except Exception:
            await self.session.rollback()
            raise

    async def get_experiment_events(
        self,
        experiment_id: str,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[ExperimentEvent]:
        """获取实验事件"""
        stmt = select(ExperimentEvent).where(ExperimentEvent.experiment_id == experiment_id)
        if event_type:
            stmt = stmt.where(ExperimentEvent.event_type == event_type)
        if start_date:
            stmt = stmt.where(ExperimentEvent.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(ExperimentEvent.timestamp <= end_date)

        result = await self.session.execute(
            stmt.order_by(desc(ExperimentEvent.timestamp)).limit(limit)
        )
        return result.scalars().all()

    async def count_experiment_events(self, experiment_id: str, event_type: Optional[str] = None) -> int:
        """统计实验事件数量"""
        stmt = select(func.count(ExperimentEvent.id)).where(ExperimentEvent.experiment_id == experiment_id)
        if event_type:
            stmt = stmt.where(ExperimentEvent.event_type == event_type)
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)

    async def get_variant_event_stats(self, experiment_id: str, event_type: str) -> Dict[str, Dict[str, float]]:
        """获取变体事件统计"""
        result = await self.session.execute(
            select(
                ExperimentEvent.variant_id,
                func.count(ExperimentEvent.id).label('count'),
                func.sum(ExperimentEvent.event_value).label('total_value'),
                func.avg(ExperimentEvent.event_value).label('avg_value'),
            )
            .where(
                and_(
                    ExperimentEvent.experiment_id == experiment_id,
                    ExperimentEvent.event_type == event_type,
                )
            )
            .group_by(ExperimentEvent.variant_id)
        )

        stats: Dict[str, Dict[str, float]] = {}
        for row in result.all():
            stats[row.variant_id] = {
                'count': row.count,
                'total_value': float(row.total_value or 0),
                'avg_value': float(row.avg_value or 0),
            }
        return stats

    async def mark_events_processed(self, event_ids: List[str]) -> int:
        """标记事件已处理"""
        try:
            result = await self.session.execute(
                update(ExperimentEvent)
                .where(ExperimentEvent.id.in_(event_ids))
                .values(processed=True)
            )
            await self.session.commit()
            return result.rowcount or 0
        except Exception:
            await self.session.rollback()
            return 0

class ExperimentMetricResultRepository(BaseRepository[ExperimentMetricResult, None]):
    """实验指标结果数据访问层"""

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, ExperimentMetricResult)

    async def save_metric_result(
        self,
        experiment_id: str,
        metric_name: str,
        result_data: Dict[str, Any],
    ) -> ExperimentMetricResult:
        """保存指标分析结果"""
        try:
            result_record = ExperimentMetricResult(
                experiment_id=experiment_id,
                metric_name=metric_name,
                variant_results=result_data['variant_results'],
                statistical_test=result_data['statistical_test'],
                p_value=result_data['p_value'],
                is_significant=result_data['is_significant'],
                effect_size=result_data['effect_size'],
                confidence_interval_lower=result_data['confidence_interval'][0],
                confidence_interval_upper=result_data['confidence_interval'][1],
                sample_size=result_data['sample_size'],
                statistical_power=result_data['statistical_power'],
                data_window_start=result_data['data_window_start'],
                data_window_end=result_data['data_window_end'],
            )
            self.session.add(result_record)
            await self.session.commit()
            await self.session.refresh(result_record)
            return result_record
        except Exception:
            await self.session.rollback()
            raise

    async def get_latest_results(self, experiment_id: str) -> List[ExperimentMetricResult]:
        """获取实验的最新指标结果"""
        subquery = (
            select(
                ExperimentMetricResult.metric_name,
                func.max(ExperimentMetricResult.computed_at).label('latest_computed_at'),
            )
            .where(ExperimentMetricResult.experiment_id == experiment_id)
            .group_by(ExperimentMetricResult.metric_name)
            .subquery()
        )

        stmt = (
            select(ExperimentMetricResult)
            .join(
                subquery,
                and_(
                    ExperimentMetricResult.metric_name == subquery.c.metric_name,
                    ExperimentMetricResult.computed_at == subquery.c.latest_computed_at,
                ),
            )
            .where(ExperimentMetricResult.experiment_id == experiment_id)
        )

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_metric_history(self, experiment_id: str, metric_name: str) -> List[ExperimentMetricResult]:
        """获取指标的历史计算结果"""
        result = await self.session.execute(
            select(ExperimentMetricResult)
            .where(
                and_(
                    ExperimentMetricResult.experiment_id == experiment_id,
                    ExperimentMetricResult.metric_name == metric_name,
                )
            )
            .order_by(desc(ExperimentMetricResult.computed_at))
        )
        return result.scalars().all()
