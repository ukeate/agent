"""
A/B测试实验平台数据访问层
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from models.database.experiment import (
    Experiment, ExperimentVariant, ExperimentAssignment, 
    ExperimentEvent, ExperimentMetricResult, ExperimentLayerConflict
)
from models.schemas.experiment import (
    ExperimentConfig, CreateExperimentRequest, UpdateExperimentRequest,
    ExperimentStatus, ExperimentAssignmentRequest, RecordEventRequest
)
from repositories.base import BaseRepository


class ExperimentRepository(BaseRepository[Experiment, ExperimentConfig]):
    """实验数据访问层"""
    
    def __init__(self, db_session: Session):
        super().__init__(Experiment, db_session)
    
    def create_experiment(self, experiment_request: CreateExperimentRequest) -> Experiment:
        """创建新实验"""
        try:
            # 创建实验记录
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
                targeting_rules=[rule.dict() for rule in experiment_request.targeting_rules],
                metadata=experiment_request.metadata.copy()
            )
            
            self.db.add(experiment)
            self.db.flush()  # 获取生成的ID
            
            # 创建变体记录
            for variant in experiment_request.variants:
                # 查找对应的流量分配
                traffic_allocation = next(
                    (alloc for alloc in experiment_request.traffic_allocation 
                     if alloc.variant_id == variant.variant_id), None
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
                    traffic_percentage=traffic_allocation.percentage
                )
                self.db.add(variant_record)
            
            self.db.commit()
            return experiment
            
        except IntegrityError as e:
            self.db.rollback()
            raise ValueError(f"Failed to create experiment: {str(e)}")
    
    def get_experiment_with_variants(self, experiment_id: str) -> Optional[Experiment]:
        """获取实验及其变体信息"""
        return (self.db.query(Experiment)
                .options(joinedload(Experiment.variants))
                .filter(Experiment.id == experiment_id)
                .first())
    
    def get_experiments_by_owner(self, owner: str, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """获取用户的实验列表"""
        query = self.db.query(Experiment).filter(Experiment.owner == owner)
        
        if status:
            query = query.filter(Experiment.status == status)
            
        return query.order_by(desc(Experiment.created_at)).all()
    
    def get_running_experiments(self, layer: Optional[str] = None) -> List[Experiment]:
        """获取运行中的实验"""
        query = (self.db.query(Experiment)
                .filter(Experiment.status == ExperimentStatus.RUNNING))
        
        if layer:
            query = query.filter(Experiment.layers.op('?')(layer))
        
        return query.all()
    
    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> bool:
        """更新实验状态"""
        try:
            rows_affected = (self.db.query(Experiment)
                           .filter(Experiment.id == experiment_id)
                           .update({
                               Experiment.status: status,
                               Experiment.updated_at: datetime.utcnow()
                           }))
            self.db.commit()
            return rows_affected > 0
        except Exception:
            self.db.rollback()
            return False
    
    def update_experiment(self, experiment_id: str, update_request: UpdateExperimentRequest) -> Optional[Experiment]:
        """更新实验配置"""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return None
            
            # 只允许更新草稿状态的实验的核心配置
            if experiment.status != ExperimentStatus.DRAFT:
                # 运行中的实验只能更新部分字段
                allowed_fields = ['end_date', 'targeting_rules', 'metadata']
                update_data = {k: v for k, v in update_request.dict(exclude_unset=True).items() 
                              if k in allowed_fields and v is not None}
            else:
                # 草稿状态可以更新所有字段
                update_data = {k: v for k, v in update_request.dict(exclude_unset=True).items() 
                              if v is not None}
            
            if update_data:
                # 处理特殊字段
                if 'targeting_rules' in update_data:
                    update_data['targeting_rules'] = [rule.dict() for rule in update_data['targeting_rules']]
                
                update_data['updated_at'] = datetime.utcnow()
                
                (self.db.query(Experiment)
                 .filter(Experiment.id == experiment_id)
                 .update(update_data))
                
                self.db.commit()
                
                # 返回更新后的实验
                return self.get_experiment_with_variants(experiment_id)
            
            return experiment
            
        except Exception:
            self.db.rollback()
            return None
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验（仅草稿状态）"""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return False
            
            # 只允许删除草稿状态的实验
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Only draft experiments can be deleted")
            
            self.db.delete(experiment)
            self.db.commit()
            return True
            
        except Exception:
            self.db.rollback()
            return False
    
    def check_layer_conflicts(self, experiment_id: str, layers: List[str]) -> List[Dict[str, Any]]:
        """检查实验层冲突"""
        conflicts = []
        
        for layer in layers:
            # 查找同一层中其他运行中的实验
            conflicting_experiments = (
                self.db.query(Experiment)
                .filter(
                    and_(
                        Experiment.id != experiment_id,
                        Experiment.status.in_([ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]),
                        Experiment.layers.op('?')(layer)
                    )
                )
                .all()
            )
            
            for conflict_exp in conflicting_experiments:
                conflicts.append({
                    'layer': layer,
                    'conflicting_experiment_id': conflict_exp.id,
                    'conflicting_experiment_name': conflict_exp.name,
                    'conflict_type': 'mutual_exclusion'
                })
        
        return conflicts


class ExperimentAssignmentRepository(BaseRepository[ExperimentAssignment, None]):
    """实验分配数据访问层"""
    
    def __init__(self, db_session: Session):
        super().__init__(ExperimentAssignment, db_session)
    
    def get_user_assignment(self, experiment_id: str, user_id: str) -> Optional[ExperimentAssignment]:
        """获取用户的实验分配"""
        return (self.db.query(ExperimentAssignment)
                .filter(
                    and_(
                        ExperimentAssignment.experiment_id == experiment_id,
                        ExperimentAssignment.user_id == user_id
                    )
                )
                .first())
    
    def create_assignment(self, experiment_id: str, user_id: str, variant_id: str, 
                         context: Dict[str, Any], is_eligible: bool = True,
                         assignment_reason: str = "traffic_split") -> ExperimentAssignment:
        """创建用户实验分配"""
        try:
            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant_id,
                context=context,
                is_eligible=is_eligible,
                assignment_reason=assignment_reason
            )
            
            self.db.add(assignment)
            self.db.commit()
            return assignment
            
        except IntegrityError:
            self.db.rollback()
            # 如果已存在分配，返回现有的
            return self.get_user_assignment(experiment_id, user_id)
    
    def get_experiment_assignments(self, experiment_id: str, limit: int = 1000) -> List[ExperimentAssignment]:
        """获取实验的所有分配"""
        return (self.db.query(ExperimentAssignment)
                .filter(ExperimentAssignment.experiment_id == experiment_id)
                .order_by(desc(ExperimentAssignment.timestamp))
                .limit(limit)
                .all())
    
    def get_variant_assignment_counts(self, experiment_id: str) -> Dict[str, int]:
        """获取各变体的分配数量"""
        results = (self.db.query(
                    ExperimentAssignment.variant_id,
                    func.count(ExperimentAssignment.id).label('count')
                )
                .filter(ExperimentAssignment.experiment_id == experiment_id)
                .group_by(ExperimentAssignment.variant_id)
                .all())
        
        return {result.variant_id: result.count for result in results}


class ExperimentEventRepository(BaseRepository[ExperimentEvent, None]):
    """实验事件数据访问层"""
    
    def __init__(self, db_session: Session):
        super().__init__(ExperimentEvent, db_session)
    
    def record_event(self, assignment_id: str, event_request: RecordEventRequest) -> ExperimentEvent:
        """记录实验事件"""
        try:
            # 获取分配信息
            assignment = (self.db.query(ExperimentAssignment)
                         .filter(ExperimentAssignment.id == assignment_id)
                         .first())
            
            if not assignment:
                raise ValueError(f"Assignment {assignment_id} not found")
            
            event = ExperimentEvent(
                experiment_id=assignment.experiment_id,
                assignment_id=assignment_id,
                variant_id=assignment.variant_id,
                user_id=event_request.user_id,
                event_type=event_request.event_type,
                event_value=event_request.event_value,
                metadata=event_request.metadata
            )
            
            self.db.add(event)
            self.db.commit()
            return event
            
        except Exception:
            self.db.rollback()
            raise
    
    def batch_record_events(self, assignment_id: str, events: List[RecordEventRequest]) -> List[ExperimentEvent]:
        """批量记录实验事件"""
        try:
            # 获取分配信息
            assignment = (self.db.query(ExperimentAssignment)
                         .filter(ExperimentAssignment.id == assignment_id)
                         .first())
            
            if not assignment:
                raise ValueError(f"Assignment {assignment_id} not found")
            
            event_records = []
            for event_request in events:
                event = ExperimentEvent(
                    experiment_id=assignment.experiment_id,
                    assignment_id=assignment_id,
                    variant_id=assignment.variant_id,
                    user_id=event_request.user_id,
                    event_type=event_request.event_type,
                    event_value=event_request.event_value,
                    metadata=event_request.metadata
                )
                event_records.append(event)
                self.db.add(event)
            
            self.db.commit()
            return event_records
            
        except Exception:
            self.db.rollback()
            raise
    
    def get_experiment_events(self, experiment_id: str, event_type: Optional[str] = None,
                             start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                             limit: int = 10000) -> List[ExperimentEvent]:
        """获取实验事件"""
        query = (self.db.query(ExperimentEvent)
                .filter(ExperimentEvent.experiment_id == experiment_id))
        
        if event_type:
            query = query.filter(ExperimentEvent.event_type == event_type)
        
        if start_date:
            query = query.filter(ExperimentEvent.timestamp >= start_date)
        
        if end_date:
            query = query.filter(ExperimentEvent.timestamp <= end_date)
        
        return (query.order_by(desc(ExperimentEvent.timestamp))
                .limit(limit)
                .all())
    
    def get_variant_event_stats(self, experiment_id: str, event_type: str) -> Dict[str, Dict[str, float]]:
        """获取变体事件统计"""
        results = (self.db.query(
                    ExperimentEvent.variant_id,
                    func.count(ExperimentEvent.id).label('count'),
                    func.sum(ExperimentEvent.event_value).label('total_value'),
                    func.avg(ExperimentEvent.event_value).label('avg_value')
                )
                .filter(
                    and_(
                        ExperimentEvent.experiment_id == experiment_id,
                        ExperimentEvent.event_type == event_type
                    )
                )
                .group_by(ExperimentEvent.variant_id)
                .all())
        
        stats = {}
        for result in results:
            stats[result.variant_id] = {
                'count': result.count,
                'total_value': float(result.total_value or 0),
                'avg_value': float(result.avg_value or 0)
            }
        
        return stats
    
    def mark_events_processed(self, event_ids: List[str]) -> int:
        """标记事件已处理"""
        try:
            rows_affected = (self.db.query(ExperimentEvent)
                           .filter(ExperimentEvent.id.in_(event_ids))
                           .update({'processed': True}))
            self.db.commit()
            return rows_affected
        except Exception:
            self.db.rollback()
            return 0


class ExperimentMetricResultRepository(BaseRepository[ExperimentMetricResult, None]):
    """实验指标结果数据访问层"""
    
    def __init__(self, db_session: Session):
        super().__init__(ExperimentMetricResult, db_session)
    
    def save_metric_result(self, experiment_id: str, metric_name: str, result_data: Dict[str, Any]) -> ExperimentMetricResult:
        """保存指标分析结果"""
        try:
            result = ExperimentMetricResult(
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
                data_window_end=result_data['data_window_end']
            )
            
            self.db.add(result)
            self.db.commit()
            return result
            
        except Exception:
            self.db.rollback()
            raise
    
    def get_latest_results(self, experiment_id: str) -> List[ExperimentMetricResult]:
        """获取实验的最新指标结果"""
        # 获取每个指标的最新计算结果
        subquery = (self.db.query(
                        ExperimentMetricResult.metric_name,
                        func.max(ExperimentMetricResult.computed_at).label('latest_computed_at')
                    )
                    .filter(ExperimentMetricResult.experiment_id == experiment_id)
                    .group_by(ExperimentMetricResult.metric_name)
                    .subquery())
        
        return (self.db.query(ExperimentMetricResult)
                .join(subquery, 
                      and_(
                          ExperimentMetricResult.metric_name == subquery.c.metric_name,
                          ExperimentMetricResult.computed_at == subquery.c.latest_computed_at
                      ))
                .filter(ExperimentMetricResult.experiment_id == experiment_id)
                .all())
    
    def get_metric_history(self, experiment_id: str, metric_name: str) -> List[ExperimentMetricResult]:
        """获取指标的历史计算结果"""
        return (self.db.query(ExperimentMetricResult)
                .filter(
                    and_(
                        ExperimentMetricResult.experiment_id == experiment_id,
                        ExperimentMetricResult.metric_name == metric_name
                    )
                )
                .order_by(desc(ExperimentMetricResult.computed_at))
                .all())