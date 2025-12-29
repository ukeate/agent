"""
A/B测试核心服务 - 提供实验管理、用户分配、事件追踪等核心功能
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.models.schemas.experiment import (
    ExperimentConfig, CreateExperimentRequest, ExperimentStatus,
    ExperimentAssignmentResponse, ExperimentResultsResponse, 
    ExperimentSummary, MetricResult, RecordEventRequest
)
from src.repositories.experiment_repository import (
    ExperimentRepository, ExperimentAssignmentRepository,
    ExperimentEventRepository, ExperimentMetricResultRepository
)
from src.services.traffic_splitter import TrafficSplitter
from src.services.statistical_analyzer import StatisticalAnalyzer

from src.core.logging import get_logger
logger = get_logger(__name__)

class ABTestingService:
    """A/B测试核心服务"""
    
    def __init__(
        self,
        experiment_repo: ExperimentRepository,
        assignment_repo: ExperimentAssignmentRepository,
        event_repo: ExperimentEventRepository,
        metric_repo: ExperimentMetricResultRepository
    ):
        self.experiment_repo = experiment_repo
        self.assignment_repo = assignment_repo
        self.event_repo = event_repo
        self.metric_repo = metric_repo
        self.traffic_splitter = TrafficSplitter()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    async def create_experiment(self, experiment_request: CreateExperimentRequest) -> ExperimentConfig:
        """创建新实验"""
        try:
            # 使用repository创建实验
            experiment_db = self.experiment_repo.create_experiment(experiment_request)
            
            # 转换为ExperimentConfig格式返回
            return await self._convert_db_to_config(experiment_db)
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """获取实验配置"""
        try:
            experiment_db = self.experiment_repo.get_experiment_with_variants(experiment_id)
            if not experiment_db:
                return None
            
            return await self._convert_db_to_config(experiment_db)
            
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {str(e)}")
            return None
    
    async def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> bool:
        """更新实验状态"""
        try:
            return self.experiment_repo.update_experiment_status(experiment_id, status)
        except Exception as e:
            logger.error(f"Failed to update experiment status {experiment_id}: {str(e)}")
            return False
    
    async def assign_user(self, experiment_id: str, user_id: str, 
                         context: Dict[str, Any]) -> Optional[ExperimentAssignmentResponse]:
        """为用户分配实验变体"""
        try:
            # 检查是否已有分配
            existing_assignment = self.assignment_repo.get_user_assignment(experiment_id, user_id)
            if existing_assignment:
                # 返回现有分配
                return await self._convert_assignment_to_response(existing_assignment)
            
            # 获取实验信息
            experiment = self.experiment_repo.get_experiment_with_variants(experiment_id)
            if not experiment or experiment.status != ExperimentStatus.RUNNING:
                return None
            
            # 检查用户是否符合定向条件
            is_eligible = await self._check_user_eligibility(experiment, user_id, context)
            if not is_eligible:
                return None
            
            # 进行流量分配
            variant_id = await self._assign_user_to_variant(experiment, user_id)
            if not variant_id:
                return None
            
            # 创建分配记录
            assignment = self.assignment_repo.create_assignment(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant_id,
                context=context,
                is_eligible=True,
                assignment_reason="traffic_split"
            )
            
            return await self._convert_assignment_to_response(assignment)
            
        except Exception as e:
            logger.error(f"Failed to assign user {user_id} to experiment {experiment_id}: {str(e)}")
            return None
    
    async def record_event(self, assignment_id: str, event_request: RecordEventRequest) -> bool:
        """记录实验事件"""
        try:
            event = self.event_repo.record_event(assignment_id, event_request)
            return event is not None
        except Exception as e:
            logger.error(f"Failed to record event for assignment {assignment_id}: {str(e)}")
            return False
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResultsResponse]:
        """获取实验结果"""
        try:
            # 获取实验信息
            experiment = self.experiment_repo.get_by_id(experiment_id)
            if not experiment:
                return None
            
            # 获取最新的指标分析结果
            metric_results = self.metric_repo.get_latest_results(experiment_id)

            variant_user_counts = self.assignment_repo.get_variant_assignment_counts(experiment_id)
            total_users = sum(variant_user_counts.values())
            total_events = self.event_repo.count_experiment_events(experiment_id)

            variants_performance: Dict[str, Dict[str, float]] = {
                v.variant_id: {} for v in self.experiment_repo.get_experiment_with_variants(experiment_id).variants
            }
            for metric_name in list(dict.fromkeys(experiment.success_metrics + experiment.guardrail_metrics)):
                stats = self.event_repo.get_variant_event_stats(experiment_id, metric_name)
                for variant_id, values in stats.items():
                    users = variant_user_counts.get(variant_id, 0)
                    variants_performance.setdefault(variant_id, {})[metric_name] = (
                        values.get("count", 0) / max(users, 1)
                    )
            
            # 构建实验摘要
            experiment_summary = ExperimentSummary(
                experiment_id=experiment.id,
                name=experiment.name,
                status=experiment.status,
                start_date=experiment.start_date,
                end_date=experiment.end_date,
                created_at=experiment.created_at,
                total_users=total_users,
                total_events=total_events,
                variants_performance=variants_performance,
                significant_metrics=[r.metric_name for r in metric_results if r.is_significant]
            )
            
            # 转换指标结果
            metrics = []
            for result in metric_results:
                metric = MetricResult(
                    metric_name=result.metric_name,
                    variant_results=result.variant_results,
                    statistical_test=result.statistical_test,
                    p_value=result.p_value,
                    is_significant=result.is_significant,
                    effect_size=result.effect_size,
                    confidence_interval=(result.confidence_interval_lower, result.confidence_interval_upper),
                    sample_size=result.sample_size,
                    statistical_power=result.statistical_power
                )
                metrics.append(metric)
            
            # 生成建议
            recommendations = await self._generate_recommendations(experiment, metrics)
            
            return ExperimentResultsResponse(
                experiment=experiment_summary,
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to get experiment results for {experiment_id}: {str(e)}")
            return None
    
    async def analyze_metric(self, experiment_id: str, metric_name: str) -> Optional[MetricResult]:
        """分析特定指标"""
        try:
            # 获取指标数据
            metric_data = await self._collect_metric_data(experiment_id, metric_name)
            if not metric_data:
                return None
            
            # 执行统计分析
            analysis_result = await self.statistical_analyzer.analyze_metric(
                experiment_id, metric_name, metric_data
            )
            
            # 保存分析结果
            self.metric_repo.save_metric_result(experiment_id, metric_name, analysis_result)
            
            # 构建返回结果
            return MetricResult(
                metric_name=metric_name,
                variant_results=analysis_result['variant_results'],
                statistical_test=analysis_result['statistical_test'],
                p_value=analysis_result['p_value'],
                is_significant=analysis_result['is_significant'],
                effect_size=analysis_result['effect_size'],
                confidence_interval=analysis_result['confidence_interval'],
                sample_size=analysis_result['sample_size'],
                statistical_power=analysis_result['statistical_power']
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze metric {metric_name} for experiment {experiment_id}: {str(e)}")
            return None
    
    async def get_experiment_sample_size(self, experiment_id: str) -> int:
        """获取实验当前样本量"""
        try:
            variant_counts = self.assignment_repo.get_variant_assignment_counts(experiment_id)
            return sum(variant_counts.values())
        except Exception as e:
            logger.error(f"Failed to get sample size for experiment {experiment_id}: {str(e)}")
            return 0
    
    # 内部辅助方法
    async def _convert_db_to_config(self, experiment_db) -> ExperimentConfig:
        """将数据库模型转换为ExperimentConfig"""
        from models.schemas.experiment import TrafficAllocation, ExperimentVariant, TargetingRule
        
        # 构建变体信息
        variants = []
        traffic_allocations = []
        
        for variant in experiment_db.variants:
            variants.append(ExperimentVariant(
                variant_id=variant.variant_id,
                name=variant.name,
                description=variant.description,
                config=variant.config,
                is_control=variant.is_control
            ))
            
            traffic_allocations.append(TrafficAllocation(
                variant_id=variant.variant_id,
                percentage=variant.traffic_percentage
            ))
        
        # 反序列化定向规则
        targeting_rules = []
        for rule_data in experiment_db.targeting_rules or []:
            targeting_rules.append(TargetingRule(**rule_data))
        
        return ExperimentConfig(
            experiment_id=experiment_db.id,
            name=experiment_db.name,
            description=experiment_db.description,
            hypothesis=experiment_db.hypothesis,
            owner=experiment_db.owner,
            status=experiment_db.status,
            variants=variants,
            traffic_allocation=traffic_allocations,
            start_date=experiment_db.start_date,
            end_date=experiment_db.end_date,
            success_metrics=experiment_db.success_metrics,
            guardrail_metrics=experiment_db.guardrail_metrics,
            minimum_sample_size=experiment_db.minimum_sample_size,
            significance_level=experiment_db.significance_level,
            power=experiment_db.power,
            layers=experiment_db.layers,
            targeting_rules=targeting_rules,
            metadata=experiment_db.metadata,
            created_at=experiment_db.created_at,
            updated_at=experiment_db.updated_at
        )
    
    async def _convert_assignment_to_response(self, assignment) -> ExperimentAssignmentResponse:
        """将分配记录转换为响应格式"""
        # 获取变体信息
        experiment = self.experiment_repo.get_experiment_with_variants(assignment.experiment_id)
        variant = None
        for v in experiment.variants:
            if v.variant_id == assignment.variant_id:
                variant = v
                break
        
        return ExperimentAssignmentResponse(
            experiment_id=assignment.experiment_id,
            variant_id=assignment.variant_id,
            variant_name=variant.name if variant else "Unknown",
            config=variant.config if variant else {},
            assignment_id=assignment.id,
            is_eligible=assignment.is_eligible,
            assignment_reason=assignment.assignment_reason
        )
    
    async def _check_user_eligibility(self, experiment, user_id: str, context: Dict[str, Any]) -> bool:
        """检查用户是否符合实验条件"""
        rules = experiment.targeting_rules or []
        if not rules:
            return True

        def _to_number(v):
            try:
                return float(v)
            except Exception:
                return None

        for rule in rules:
            attr = rule.get("attribute")
            op = rule.get("operator")
            expected = rule.get("value")
            actual = context.get(attr) if attr else None
            if attr is None or op is None:
                return False

            if op in {"eq", "ne"}:
                ok = actual == expected
                if op == "ne":
                    ok = not ok
            elif op in {"in", "not_in"}:
                expected_list = expected if isinstance(expected, list) else [expected]
                ok = actual in expected_list
                if op == "not_in":
                    ok = not ok
            else:
                actual_num = _to_number(actual)
                expected_num = _to_number(expected)
                if actual_num is None or expected_num is None:
                    return False
                if op == "gt":
                    ok = actual_num > expected_num
                elif op == "lt":
                    ok = actual_num < expected_num
                elif op == "gte":
                    ok = actual_num >= expected_num
                elif op == "lte":
                    ok = actual_num <= expected_num
                else:
                    return False

            if not ok:
                return False
        return True
    
    async def _assign_user_to_variant(self, experiment, user_id: str) -> Optional[str]:
        """为用户分配变体"""
        try:
            # 构建流量分配信息
            allocations = []
            for variant in experiment.variants:
                from models.schemas.experiment import TrafficAllocation
                allocations.append(TrafficAllocation(
                    variant_id=variant.variant_id,
                    percentage=variant.traffic_percentage
                ))
            
            # 使用流量分配器进行分配
            return self.traffic_splitter.get_variant(user_id, experiment.id, allocations)
            
        except Exception as e:
            logger.error(f"Failed to assign variant for user {user_id}: {str(e)}")
            return None
    
    async def _collect_metric_data(self, experiment_id: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """收集指标数据"""
        try:
            # 获取实验事件数据
            events = self.event_repo.get_experiment_events(
                experiment_id=experiment_id,
                event_type=metric_name
            )
            
            if not events:
                return None
            
            # 按变体组织数据
            variant_data = {}
            for event in events:
                variant_id = event.variant_id
                if variant_id not in variant_data:
                    variant_data[variant_id] = []
                variant_data[variant_id].append({
                    'value': event.event_value,
                    'timestamp': event.timestamp,
                    'user_id': event.user_id
                })
            
            return {
                'experiment_id': experiment_id,
                'metric_name': metric_name,
                'variant_data': variant_data,
                'total_events': len(events),
                'collection_time': utc_now()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect metric data for {metric_name}: {str(e)}")
            return None
    
    async def _generate_recommendations(self, experiment, metrics: List[MetricResult]) -> List[str]:
        """生成实验建议"""
        recommendations = []
        
        try:
            # 基于统计显著性生成建议
            significant_metrics = [m for m in metrics if m.is_significant]
            
            if not significant_metrics:
                recommendations.append("没有发现统计显著的指标差异，建议延长实验时间或增加样本量")
            else:
                # 分析显著指标的效果方向
                positive_metrics = [m for m in significant_metrics if m.effect_size > 0]
                negative_metrics = [m for m in significant_metrics if m.effect_size < 0]
                
                if positive_metrics:
                    recommendations.append(f"发现{len(positive_metrics)}个指标有显著正向提升")
                
                if negative_metrics:
                    recommendations.append(f"发现{len(negative_metrics)}个指标有显著负向影响，需要注意")
            
            # 基于样本量生成建议
            total_sample_size = sum(m.sample_size for m in metrics) / len(metrics) if metrics else 0
            if total_sample_size < experiment.minimum_sample_size:
                recommendations.append(f"当前样本量({int(total_sample_size)})未达到最小要求({experiment.minimum_sample_size})")
            
            # 基于统计功效生成建议
            low_power_metrics = [m for m in metrics if m.statistical_power < 0.8]
            if low_power_metrics:
                recommendations.append(f"{len(low_power_metrics)}个指标的统计功效不足，可能存在二类错误风险")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            recommendations.append("生成建议时发生错误，请手动分析实验结果")
        
        return recommendations
