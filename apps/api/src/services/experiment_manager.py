"""
A/B测试实验管理服务 - 实验状态机和生命周期管理
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from enum import Enum
from dataclasses import dataclass
from src.models.schemas.experiment import ExperimentStatus, CreateExperimentRequest, ExperimentConfig
from src.services.ab_testing_service import ABTestingService

from src.core.logging import get_logger
logger = get_logger(__name__)

class ExperimentTransition(Enum):
    """实验状态转换"""
    # 从 DRAFT 状态可以转换到
    DRAFT_TO_RUNNING = "draft_to_running"
    DRAFT_TO_TERMINATED = "draft_to_terminated"
    
    # 从 RUNNING 状态可以转换到
    RUNNING_TO_PAUSED = "running_to_paused"
    RUNNING_TO_COMPLETED = "running_to_completed"
    RUNNING_TO_TERMINATED = "running_to_terminated"
    
    # 从 PAUSED 状态可以转换到
    PAUSED_TO_RUNNING = "paused_to_running"
    PAUSED_TO_COMPLETED = "paused_to_completed"
    PAUSED_TO_TERMINATED = "paused_to_terminated"
    
    # 终态（不允许转换出去）
    # COMPLETED - 不允许转换
    # TERMINATED - 不允许转换

@dataclass
class StateTransitionRule:
    """状态转换规则"""
    from_status: ExperimentStatus
    to_status: ExperimentStatus
    transition: ExperimentTransition
    preconditions: List[str]  # 前置条件描述
    validations: List[str]    # 需要验证的条件

class ExperimentStateMachine:
    """实验状态机"""
    
    def __init__(self):
        self.transition_rules = {
            # DRAFT 状态的转换
            (ExperimentStatus.DRAFT, ExperimentStatus.RUNNING): StateTransitionRule(
                from_status=ExperimentStatus.DRAFT,
                to_status=ExperimentStatus.RUNNING,
                transition=ExperimentTransition.DRAFT_TO_RUNNING,
                preconditions=[
                    "实验配置完整且有效",
                    "至少有一个对照组",
                    "流量分配总和为100%",
                    "开始时间已到或未来时间",
                    "没有同层实验冲突"
                ],
                validations=[
                    "validate_experiment_config",
                    "validate_traffic_allocation",
                    "validate_layer_conflicts",
                    "validate_start_time"
                ]
            ),
            (ExperimentStatus.DRAFT, ExperimentStatus.TERMINATED): StateTransitionRule(
                from_status=ExperimentStatus.DRAFT,
                to_status=ExperimentStatus.TERMINATED,
                transition=ExperimentTransition.DRAFT_TO_TERMINATED,
                preconditions=["可以随时终止草稿实验"],
                validations=[]
            ),
            
            # RUNNING 状态的转换
            (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED): StateTransitionRule(
                from_status=ExperimentStatus.RUNNING,
                to_status=ExperimentStatus.PAUSED,
                transition=ExperimentTransition.RUNNING_TO_PAUSED,
                preconditions=["实验正在运行"],
                validations=["validate_running_experiment"]
            ),
            (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED): StateTransitionRule(
                from_status=ExperimentStatus.RUNNING,
                to_status=ExperimentStatus.COMPLETED,
                transition=ExperimentTransition.RUNNING_TO_COMPLETED,
                preconditions=[
                    "达到最小样本量",
                    "运行时间满足要求",
                    "统计显著性检验完成"
                ],
                validations=[
                    "validate_minimum_sample_size",
                    "validate_experiment_duration",
                    "validate_statistical_significance"
                ]
            ),
            (ExperimentStatus.RUNNING, ExperimentStatus.TERMINATED): StateTransitionRule(
                from_status=ExperimentStatus.RUNNING,
                to_status=ExperimentStatus.TERMINATED,
                transition=ExperimentTransition.RUNNING_TO_TERMINATED,
                preconditions=["可以随时终止运行中的实验"],
                validations=["validate_termination_reason"]
            ),
            
            # PAUSED 状态的转换
            (ExperimentStatus.PAUSED, ExperimentStatus.RUNNING): StateTransitionRule(
                from_status=ExperimentStatus.PAUSED,
                to_status=ExperimentStatus.RUNNING,
                transition=ExperimentTransition.PAUSED_TO_RUNNING,
                preconditions=["实验处于暂停状态", "没有阻塞问题"],
                validations=["validate_resume_conditions"]
            ),
            (ExperimentStatus.PAUSED, ExperimentStatus.COMPLETED): StateTransitionRule(
                from_status=ExperimentStatus.PAUSED,
                to_status=ExperimentStatus.COMPLETED,
                transition=ExperimentTransition.PAUSED_TO_COMPLETED,
                preconditions=[
                    "达到最小样本量",
                    "统计显著性检验完成"
                ],
                validations=[
                    "validate_minimum_sample_size",
                    "validate_statistical_significance"
                ]
            ),
            (ExperimentStatus.PAUSED, ExperimentStatus.TERMINATED): StateTransitionRule(
                from_status=ExperimentStatus.PAUSED,
                to_status=ExperimentStatus.TERMINATED,
                transition=ExperimentTransition.PAUSED_TO_TERMINATED,
                preconditions=["可以随时终止暂停中的实验"],
                validations=["validate_termination_reason"]
            ),
        }
    
    def can_transition(self, from_status: ExperimentStatus, to_status: ExperimentStatus) -> bool:
        """检查是否可以进行状态转换"""
        return (from_status, to_status) in self.transition_rules
    
    def get_transition_rule(self, from_status: ExperimentStatus, 
                           to_status: ExperimentStatus) -> Optional[StateTransitionRule]:
        """获取状态转换规则"""
        return self.transition_rules.get((from_status, to_status))
    
    def get_valid_transitions(self, from_status: ExperimentStatus) -> List[ExperimentStatus]:
        """获取某状态可以转换到的所有状态"""
        valid_transitions = []
        for (from_state, to_state) in self.transition_rules.keys():
            if from_state == from_status:
                valid_transitions.append(to_state)
        return valid_transitions

class ExperimentManager:
    """实验管理器 - 负责实验的生命周期管理"""
    
    def __init__(self, ab_testing_service: ABTestingService):
        self.ab_service = ab_testing_service
        self.state_machine = ExperimentStateMachine()
    
    async def create_experiment(self, experiment_request: CreateExperimentRequest) -> ExperimentConfig:
        """创建新实验"""
        logger.info(f"Creating experiment: {experiment_request.name}")
        
        try:
            # 验证实验配置
            await self._validate_experiment_config(experiment_request)
            
            # 检查层冲突
            await self._check_layer_conflicts(None, experiment_request.layers)
            
            # 创建实验
            experiment = await self.ab_service.create_experiment(experiment_request)
            
            logger.info(f"Successfully created experiment {experiment.experiment_id}")
            return experiment
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """启动实验"""
        logger.info(f"Starting experiment {experiment_id}")
        
        try:
            # 获取实验
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # 检查状态转换是否允许
            if not self.state_machine.can_transition(experiment.status, ExperimentStatus.RUNNING):
                raise ValueError(f"Cannot start experiment from status {experiment.status}")
            
            # 执行转换前验证
            await self._validate_transition(experiment, ExperimentStatus.RUNNING)
            
            # 更新状态
            success = await self.ab_service.update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            if not success:
                raise RuntimeError("Failed to update experiment status")
            
            # 执行启动后的处理
            await self._on_experiment_started(experiment_id)
            
            # 返回更新后的实验
            return await self.ab_service.get_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {str(e)}")
            raise
    
    async def pause_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """暂停实验"""
        logger.info(f"Pausing experiment {experiment_id}")
        
        try:
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if not self.state_machine.can_transition(experiment.status, ExperimentStatus.PAUSED):
                raise ValueError(f"Cannot pause experiment from status {experiment.status}")
            
            await self._validate_transition(experiment, ExperimentStatus.PAUSED)
            
            success = await self.ab_service.update_experiment_status(experiment_id, ExperimentStatus.PAUSED)
            if not success:
                raise RuntimeError("Failed to update experiment status")
            
            await self._on_experiment_paused(experiment_id)
            
            return await self.ab_service.get_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to pause experiment {experiment_id}: {str(e)}")
            raise
    
    async def resume_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """恢复暂停的实验"""
        logger.info(f"Resuming experiment {experiment_id}")
        
        try:
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment.status != ExperimentStatus.PAUSED:
                raise ValueError(f"Experiment is not paused, current status: {experiment.status}")
            
            await self._validate_transition(experiment, ExperimentStatus.RUNNING)
            
            success = await self.ab_service.update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            if not success:
                raise RuntimeError("Failed to update experiment status")
            
            await self._on_experiment_resumed(experiment_id)
            
            return await self.ab_service.get_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to resume experiment {experiment_id}: {str(e)}")
            raise
    
    async def stop_experiment(self, experiment_id: str, reason: str = "manual_stop") -> Optional[ExperimentConfig]:
        """停止实验（完成状态）"""
        logger.info(f"Stopping experiment {experiment_id}, reason: {reason}")
        
        try:
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # 检查是否可以正常完成
            if self.state_machine.can_transition(experiment.status, ExperimentStatus.COMPLETED):
                await self._validate_transition(experiment, ExperimentStatus.COMPLETED)
                target_status = ExperimentStatus.COMPLETED
                
                # 执行完成前的处理
                await self._on_experiment_completing(experiment_id)
                
            else:
                # 强制终止
                target_status = ExperimentStatus.TERMINATED
                await self._on_experiment_terminating(experiment_id, reason)
            
            success = await self.ab_service.update_experiment_status(experiment_id, target_status)
            if not success:
                raise RuntimeError("Failed to update experiment status")
            
            if target_status == ExperimentStatus.COMPLETED:
                await self._on_experiment_completed(experiment_id)
            else:
                await self._on_experiment_terminated(experiment_id, reason)
            
            return await self.ab_service.get_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {str(e)}")
            raise
    
    async def terminate_experiment(self, experiment_id: str, reason: str) -> Optional[ExperimentConfig]:
        """终止实验（异常终止）"""
        logger.info(f"Terminating experiment {experiment_id}, reason: {reason}")
        
        try:
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # 终止可以从任何非终态进行
            if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.TERMINATED]:
                raise ValueError(f"Experiment is already in final state: {experiment.status}")
            
            await self._on_experiment_terminating(experiment_id, reason)
            
            success = await self.ab_service.update_experiment_status(experiment_id, ExperimentStatus.TERMINATED)
            if not success:
                raise RuntimeError("Failed to update experiment status")
            
            await self._on_experiment_terminated(experiment_id, reason)
            
            return await self.ab_service.get_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to terminate experiment {experiment_id}: {str(e)}")
            raise
    
    async def get_experiment_status_transitions(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验可进行的状态转换"""
        try:
            experiment = await self.ab_service.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            current_status = experiment.status
            valid_transitions = self.state_machine.get_valid_transitions(current_status)
            
            transitions = {}
            for target_status in valid_transitions:
                rule = self.state_machine.get_transition_rule(current_status, target_status)
                transitions[target_status.value] = {
                    "transition_name": rule.transition.value,
                    "preconditions": rule.preconditions,
                    "is_valid": await self._can_perform_transition(experiment, target_status)
                }
            
            return {
                "experiment_id": experiment_id,
                "current_status": current_status.value,
                "available_transitions": transitions
            }
            
        except Exception as e:
            logger.error(f"Failed to get status transitions for experiment {experiment_id}: {str(e)}")
            raise
    
    # 生命周期事件处理
    async def on_experiment_started(self, experiment_id: str):
        """实验启动后的处理（后台任务）"""
        await self._on_experiment_started(experiment_id)
    
    async def _validate_experiment_config(self, experiment_request: CreateExperimentRequest):
        """验证实验配置"""
        # 验证变体配置
        if len(experiment_request.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        # 验证对照组
        control_groups = [v for v in experiment_request.variants if v.is_control]
        if len(control_groups) != 1:
            raise ValueError("Experiment must have exactly one control group")
        
        # 验证流量分配
        total_allocation = sum(allocation.percentage for allocation in experiment_request.traffic_allocation)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_allocation}%")
        
        # 验证成功指标
        if not experiment_request.success_metrics:
            raise ValueError("Experiment must have at least one success metric")
    
    async def _check_layer_conflicts(self, experiment_id: Optional[str], layers: List[str]):
        """检查层冲突"""
        if not layers:
            return

        conflicts = self.ab_service.experiment_repo.check_layer_conflicts(experiment_id or "", layers)
        if conflicts:
            detail = "; ".join(
                f"layer={c.get('layer')} conflict_experiment={c.get('conflicting_experiment_id')}"
                for c in conflicts
            )
            raise ValueError(f"检测到实验层冲突: {detail}")
    
    async def _validate_transition(self, experiment: ExperimentConfig, target_status: ExperimentStatus):
        """验证状态转换条件"""
        rule = self.state_machine.get_transition_rule(experiment.status, target_status)
        if not rule:
            raise ValueError(f"Invalid transition from {experiment.status} to {target_status}")
        
        # 执行验证方法
        for validation in rule.validations:
            await self._execute_validation(validation, experiment, target_status)
    
    async def _execute_validation(self, validation: str, experiment: ExperimentConfig, target_status: ExperimentStatus):
        """执行具体的验证逻辑"""
        if validation == "validate_experiment_config":
            # 验证实验配置完整性
            if len(experiment.variants) < 2:
                raise ValueError("实验至少需要2个变体")
            if len([v for v in experiment.variants if v.is_control]) != 1:
                raise ValueError("实验必须且只能有一个对照组")
            if not experiment.success_metrics:
                raise ValueError("实验至少需要一个成功指标")
        elif validation == "validate_traffic_allocation":
            # 验证流量分配
            total_allocation = sum(a.percentage for a in experiment.traffic_allocation)
            if abs(total_allocation - 100.0) > 0.01:
                raise ValueError(f"流量分配总和必须为100%，当前为 {total_allocation}%")
        elif validation == "validate_layer_conflicts":
            await self._check_layer_conflicts(experiment.experiment_id, experiment.layers)
        elif validation == "validate_start_time":
            # 验证开始时间
            if experiment.start_date > utc_now() + timedelta(days=1):
                raise ValueError("Start date is too far in the future")
        elif validation == "validate_running_experiment":
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError("实验未处于运行状态")
        elif validation == "validate_minimum_sample_size":
            # 验证最小样本量
            current_sample_size = await self.ab_service.get_experiment_sample_size(experiment.experiment_id)
            if current_sample_size < experiment.minimum_sample_size:
                raise ValueError(f"Sample size {current_sample_size} is below minimum {experiment.minimum_sample_size}")
        elif validation == "validate_experiment_duration":
            if experiment.end_date and utc_now() < experiment.end_date:
                raise ValueError("实验尚未到达结束时间，无法完成")
        elif validation == "validate_statistical_significance":
            # 验证统计显著性
            results = await self.ab_service.get_experiment_results(experiment.experiment_id)
            if results and not any(metric.is_significant for metric in results.metrics):
                logger.warning(f"No statistically significant results found for experiment {experiment.experiment_id}")
        elif validation == "validate_resume_conditions":
            if experiment.end_date and utc_now() > experiment.end_date:
                raise ValueError("实验已超过结束时间，无法恢复")
        elif validation == "validate_termination_reason":
            return
        else:
            raise ValueError(f"未知的验证项: {validation}")
        # 添加其他验证逻辑...
    
    async def _can_perform_transition(self, experiment: ExperimentConfig, target_status: ExperimentStatus) -> bool:
        """检查是否可以执行状态转换"""
        try:
            await self._validate_transition(experiment, target_status)
            return True
        except Exception:
            return False
    
    # 状态转换事件处理
    async def _on_experiment_started(self, experiment_id: str):
        """实验启动后的处理"""
        logger.info(f"Processing experiment start event for {experiment_id}")
    
    async def _on_experiment_paused(self, experiment_id: str):
        """实验暂停后的处理"""
        logger.info(f"Processing experiment pause event for {experiment_id}")
    
    async def _on_experiment_resumed(self, experiment_id: str):
        """实验恢复后的处理"""
        logger.info(f"Processing experiment resume event for {experiment_id}")
    
    async def _on_experiment_completing(self, experiment_id: str):
        """实验完成前的处理"""
        logger.info(f"Processing experiment completing event for {experiment_id}")
        experiment = await self.ab_service.get_experiment(experiment_id)
        if not experiment:
            return
        metrics = list(dict.fromkeys(experiment.success_metrics + experiment.guardrail_metrics))
        for metric_name in metrics:
            await self.ab_service.analyze_metric(experiment_id, metric_name)
    
    async def _on_experiment_completed(self, experiment_id: str):
        """实验完成后的处理"""
        logger.info(f"Processing experiment completed event for {experiment_id}")
    
    async def _on_experiment_terminating(self, experiment_id: str, reason: str):
        """实验终止前的处理"""
        logger.info(f"Processing experiment terminating event for {experiment_id}, reason: {reason}")
    
    async def _on_experiment_terminated(self, experiment_id: str, reason: str):
        """实验终止后的处理"""
        logger.info(f"Processing experiment terminated event for {experiment_id}, reason: {reason}")
