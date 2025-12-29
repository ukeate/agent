"""
实验管理系统

提供实验生命周期管理，包括创建、启动、监控、停止和删除实验的功能。
支持异步操作和后台任务处理。
"""

from fastapi import HTTPException
from typing import Dict, List, Optional, Any, Callable, AsyncContextManager
import uuid as uuid_lib
from src.core.utils.timezone_utils import utc_now
import json
import asyncio
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from .models import (
    ExperimentModel,
    TrialModel,
    StudyMetadataModel,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentDetail,
    TrialResponse
)
from .optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    HyperparameterRange,
    OptimizationAlgorithm,
    PruningAlgorithm,
    ResourceManager
)
from .search_engine import HyperparameterSearchEngine
from src.core.database import get_db_session

from src.core.logging import get_logger

class ExperimentManager:
    """实验管理器"""
    
    def __init__(
        self,
        session_factory: Callable[[], AsyncContextManager[AsyncSession]] = get_db_session,
    ):
        self.session_factory = session_factory
        self.active_experiments = {}
        self.resource_manager = ResourceManager()
        self.search_engine = HyperparameterSearchEngine()
        self.logger = get_logger(__name__)
    
    async def create_experiment(self, request: ExperimentRequest) -> ExperimentResponse:
        """异步创建实验"""
        async with self.session_factory() as db:
            try:
                if not request.parameters:
                    raise HTTPException(status_code=400, detail="Parameters cannot be empty")
                
                experiment = ExperimentModel(
                    name=request.name,
                    description=request.description,
                    algorithm=request.algorithm,
                    objective=request.objective,
                    config=json.dumps({
                        "n_trials": request.n_trials,
                        "timeout": request.timeout,
                        "early_stopping": request.early_stopping,
                        "patience": request.patience,
                        "min_improvement": request.min_improvement,
                        "max_concurrent_trials": request.max_concurrent_trials
                    }),
                    parameters=json.dumps([param.model_dump(mode="json") for param in request.parameters]),
                    status="created",
                    max_concurrent_trials=request.max_concurrent_trials
                )
                
                db.add(experiment)
                await db.commit()
                await db.refresh(experiment)
                
                study_metadata = StudyMetadataModel(
                    study_name=f"exp_{experiment.id}",
                    experiment_id=experiment.id,
                    search_space=json.dumps([param.model_dump(mode="json") for param in request.parameters])
                )
                
                db.add(study_metadata)
                await db.commit()
                
                self.logger.info(f"Created experiment: {experiment.id}")
                
                return ExperimentResponse(
                    id=str(experiment.id),
                    name=experiment.name,
                    status=experiment.status,
                    algorithm=experiment.algorithm,
                    objective=experiment.objective,
                    created_at=experiment.created_at
                )
                
            except SQLAlchemyError as e:
                await db.rollback()
                self.logger.error(f"Database error creating experiment: {e}")
                raise HTTPException(status_code=500, detail="Database error")
    
    async def list_experiments(self, skip: int = 0, limit: int = 100) -> List[ExperimentResponse]:
        """异步列出实验"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel)
                .order_by(ExperimentModel.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            experiments = result.scalars().all()
            return [
                ExperimentResponse(
                    id=str(exp.id),
                    name=exp.name,
                    status=exp.status,
                    algorithm=exp.algorithm,
                    objective=exp.objective,
                    created_at=exp.created_at,
                )
                for exp in experiments
            ]
    
    async def get_experiment(self, experiment_id: str) -> ExperimentDetail:
        """异步获取实验详情"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            trial_result = await db.execute(
                select(TrialModel).where(TrialModel.experiment_id == experiment.id)
            )
            trials = trial_result.scalars().all()
            
            return ExperimentDetail(
                id=str(experiment.id),
                name=experiment.name,
                description=experiment.description,
                status=experiment.status,
                algorithm=experiment.algorithm,
                objective=experiment.objective,
                config=json.loads(experiment.config) if experiment.config else {},
                parameters=json.loads(experiment.parameters) if experiment.parameters else [],
                best_value=experiment.best_value,
                best_params=json.loads(experiment.best_params) if experiment.best_params else {},
                total_trials=experiment.total_trials,
                successful_trials=experiment.successful_trials,
                pruned_trials=experiment.pruned_trials,
                failed_trials=experiment.failed_trials,
                created_at=experiment.created_at,
                started_at=experiment.started_at,
                completed_at=experiment.completed_at,
                trials_count=len(trials),
            )
    
    async def start_experiment(
        self, 
        experiment_id: str, 
        objective_function: Optional[Callable[[Dict[str, Any]], float]] = None
    ):
        """异步启动实验"""
        async with self.session_factory() as db:
            experiment: ExperimentModel | None = None
            try:
                result = await db.execute(
                    select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
                )
                experiment = result.scalar_one_or_none()
                
                if not experiment:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                if experiment.status != "created":
                    raise HTTPException(status_code=400, detail="Experiment already started or completed")
                
                if not self.resource_manager.can_start_trial():
                    raise HTTPException(status_code=503, detail="Insufficient resources to start experiment")
                
                experiment.status = "running"
                experiment.started_at = utc_now()
                await db.commit()
                
                resource_requirement = {
                    "gpu": 1.0,
                    "cpu": experiment.max_concurrent_trials or 4,
                    "memory": 8.0
                }
                self.resource_manager.register_trial(experiment_id, resource_requirement)
                
                self.active_experiments[experiment_id] = {
                    "experiment": experiment,
                    "start_time": utc_now()
                }
                
                asyncio.create_task(self._run_optimization(experiment_id, objective_function))
                
                self.logger.info(f"Started experiment: {experiment_id}")
                
            except Exception as e:
                if experiment is not None:
                    experiment.status = "failed"
                    await db.commit()
                self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
                raise
    
    async def _run_optimization(
        self, 
        experiment_id: str, 
        objective_function: Optional[Callable[[Dict[str, Any]], float]]
    ):
        """运行优化（后台任务）"""
        async with self.session_factory() as db:
            experiment: ExperimentModel | None = None
            try:
                result = await db.execute(
                    select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
                )
                experiment = result.scalar_one_or_none()
                
                if not experiment:
                    return
                
                config_data = json.loads(experiment.config)
                params_data = json.loads(experiment.parameters)
                
                optimization_config = OptimizationConfig(
                    study_name=f"exp_{experiment_id}",
                    algorithm=OptimizationAlgorithm(experiment.algorithm),
                    pruning=PruningAlgorithm.HYPERBAND,
                    direction=experiment.objective,
                    n_trials=config_data.get("n_trials", 100),
                    timeout=config_data.get("timeout"),
                    early_stopping=config_data.get("early_stopping", True),
                    patience=config_data.get("patience", 20),
                    min_improvement=config_data.get("min_improvement", 0.001),
                    max_concurrent_trials=config_data.get("max_concurrent_trials", 5),
                )
                
                parameter_ranges = [HyperparameterRange(**param_data) for param_data in params_data]
                
                optimizer = HyperparameterOptimizer(optimization_config)
                for param_range in parameter_ranges:
                    optimizer.add_parameter_range(param_range)
                
                if objective_function is None:
                    def default_objective(params: Dict[str, Any]) -> float:
                        """默认目标函数 - 确定性Rosenbrock函数"""
                        total = 0.0
                        param_values = [
                            v for v in params.values() if isinstance(v, (int, float))
                        ]
                        if len(param_values) == 1:
                            x = param_values[0]
                            total = (1 - x) ** 2
                        else:
                            for i in range(len(param_values) - 1):
                                x, y = param_values[i], param_values[i + 1]
                                total += 100 * (y - x**2) ** 2 + (1 - x) ** 2
                        return -total
                    
                    objective_function = default_objective
                
                loop = asyncio.get_running_loop()
                def progress_callback(study, trial):
                    asyncio.run_coroutine_threadsafe(
                        self._save_trial_to_db(experiment_id, trial),
                        loop
                    )
                
                result = await asyncio.to_thread(
                    optimizer.optimize,
                    objective_function,
                    callbacks=[progress_callback]
                )
                
                await db.refresh(experiment)
                experiment.status = "completed"
                experiment.completed_at = utc_now()
                experiment.best_value = result["best_value"]
                experiment.best_params = json.dumps(result["best_params"])
                experiment.total_trials = result["stats"]["total_trials"]
                experiment.successful_trials = result["stats"]["successful_trials"]
                experiment.pruned_trials = result["stats"]["pruned_trials"]
                
                await db.commit()
                
                try:
                    visualizations = optimizer.create_visualizations(
                        save_path=f"./optimization_results/{experiment_id}"
                    )
                    self.logger.info(f"Created {len(visualizations)} visualizations for experiment {experiment_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create visualizations: {e}")
                
                self.logger.info(f"Experiment {experiment_id} completed successfully")
                
            except Exception as e:
                if experiment is not None:
                    await db.refresh(experiment)
                    experiment.status = "failed"
                    experiment.completed_at = utc_now()
                    await db.commit()
                self.logger.error(f"Experiment {experiment_id} failed: {e}")
            finally:
                self.resource_manager.unregister_trial(experiment_id)
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]
    
    async def _save_trial_to_db(self, experiment_id: str, trial):
        """保存试验到数据库"""
        async with self.session_factory() as db:
            try:
                existing_result = await db.execute(
                    select(TrialModel).where(
                        TrialModel.experiment_id == uuid_lib.UUID(experiment_id),
                        TrialModel.trial_number == trial.number,
                    )
                )
                existing_trial = existing_result.scalar_one_or_none()
                
                if existing_trial:
                    existing_trial.value = trial.value
                    existing_trial.state = trial.state.name
                    existing_trial.end_time = trial.datetime_complete
                    existing_trial.duration = trial.duration.total_seconds() if trial.duration else None
                else:
                    trial_record = TrialModel(
                        experiment_id=uuid_lib.UUID(experiment_id),
                        trial_number=trial.number,
                        parameters=json.dumps(trial.params),
                        value=trial.value,
                        state=trial.state.name,
                        start_time=trial.datetime_start,
                        end_time=trial.datetime_complete,
                        duration=trial.duration.total_seconds() if trial.duration else None,
                    )
                    db.add(trial_record)
                
                await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to save trial to database: {e}")
    
    async def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """异步停止实验"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            if experiment.status != "running":
                raise HTTPException(status_code=400, detail="Experiment is not running")
            
            experiment.status = "stopped"
            experiment.completed_at = utc_now()
            await db.commit()
            
            self.resource_manager.unregister_trial(experiment_id)
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            self.logger.info(f"Stopped experiment: {experiment_id}")
            
            return {"status": "stopped", "experiment_id": experiment_id}
    
    async def delete_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """异步删除实验"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()
            
            if experiment and experiment.status == "running":
                await self.stop_experiment(experiment_id)
            
            deleted_count = await db.execute(
                ExperimentModel.__table__.delete().where(
                    ExperimentModel.id == uuid_lib.UUID(experiment_id)
                )
            )
            
            if not deleted_count.rowcount:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            await db.execute(
                StudyMetadataModel.__table__.delete().where(
                    StudyMetadataModel.experiment_id == uuid_lib.UUID(experiment_id)
                )
            )
            
            await db.commit()
            
            self.logger.info(f"Deleted experiment: {experiment_id}")
            
            return {"status": "deleted", "experiment_id": experiment_id}
    
    async def get_trials(self, experiment_id: str) -> List[TrialResponse]:
        """异步获取试验列表"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(TrialModel)
                .where(TrialModel.experiment_id == uuid_lib.UUID(experiment_id))
                .order_by(TrialModel.trial_number)
            )
            trials = result.scalars().all()
            return [
                TrialResponse(
                    id=str(trial.id),
                    trial_number=trial.trial_number,
                    parameters=json.loads(trial.parameters) if trial.parameters else {},
                    value=trial.value,
                    state=trial.state,
                    start_time=trial.start_time,
                    end_time=trial.end_time,
                    duration=trial.duration,
                    error_message=trial.error_message,
                    intermediate_values=json.loads(trial.intermediate_values) if trial.intermediate_values else {},
                    resource_usage=json.loads(trial.resource_usage) if trial.resource_usage else {}
                )
                for trial in trials
            ]
    
    async def get_visualizations(self, experiment_id: str) -> Dict[str, Any]:
        """异步获取可视化"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
        
        # 返回可视化文件路径
        base_path = f"/static/visualizations/{experiment_id}"
        
        return {
            "optimization_history": f"{base_path}/optimization_history.html",
            "param_importance": f"{base_path}/param_importance.html",
            "parallel_coordinate": f"{base_path}/parallel_coordinate.html",
            "param_slice": f"{base_path}/param_slice.html",
            "contour": f"{base_path}/contour.html"
        }

    async def update_experiment_state(self, experiment_id: str, state: Any) -> ExperimentModel | None:
        """更新实验状态"""
        status_value = state.value if hasattr(state, "value") else str(state)
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()
            if not experiment:
                return None
            experiment.status = status_value
            experiment.updated_at = utc_now()
            await db.commit()
            await db.refresh(experiment)
            return experiment

    async def create_trial(self, experiment_id: str, trial_data: Dict[str, Any]) -> TrialModel:
        """创建试验"""
        async with self.session_factory() as db:
            trial_number = trial_data.get("trial_number")
            if trial_number is None:
                result = await db.execute(
                    select(func.max(TrialModel.trial_number)).where(
                        TrialModel.experiment_id == uuid_lib.UUID(experiment_id)
                    )
                )
                max_number = result.scalar_one()
                trial_number = (max_number or 0) + 1

            state_value = trial_data.get("state")
            if hasattr(state_value, "value"):
                state_value = state_value.value

            trial = TrialModel(
                experiment_id=uuid_lib.UUID(experiment_id),
                trial_number=trial_number,
                parameters=trial_data.get("parameters"),
                value=trial_data.get("value"),
                state=state_value or "RUNNING",
                intermediate_values=trial_data.get("intermediate_values"),
                start_time=trial_data.get("start_time"),
                end_time=trial_data.get("end_time"),
                duration=trial_data.get("duration"),
                error_message=trial_data.get("error_message"),
                system_attrs=trial_data.get("system_attrs"),
                user_attrs=trial_data.get("user_attrs") or trial_data.get("metrics"),
                resource_usage=trial_data.get("resource_usage"),
            )
            db.add(trial)
            await db.commit()
            await db.refresh(trial)
            return trial

    async def update_trial_result(
        self,
        trial_id: str,
        value: Optional[float],
        state: Any,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TrialModel | None:
        """更新试验结果"""
        state_value = state.value if hasattr(state, "value") else str(state)
        async with self.session_factory() as db:
            result = await db.execute(
                select(TrialModel).where(TrialModel.id == uuid_lib.UUID(trial_id))
            )
            trial = result.scalar_one_or_none()
            if not trial:
                return None
            trial.value = value
            trial.state = state_value
            if metrics is not None:
                trial.user_attrs = metrics
            trial.end_time = trial.end_time or utc_now()
            if trial.start_time and trial.end_time and trial.duration is None:
                trial.duration = (trial.end_time - trial.start_time).total_seconds()
            await db.commit()
            await db.refresh(trial)
            return trial

    async def get_experiment_trials(self, experiment_id: str) -> List[TrialModel]:
        """获取实验的试验列表"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(TrialModel)
                .where(TrialModel.experiment_id == uuid_lib.UUID(experiment_id))
                .order_by(TrialModel.start_time)
            )
            return result.scalars().all()

    async def get_best_trial(self, experiment_id: str) -> TrialModel | None:
        """获取最佳试验"""
        async with self.session_factory() as db:
            exp_result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = exp_result.scalar_one_or_none()
            if not experiment:
                return None
            trial_result = await db.execute(
                select(TrialModel).where(TrialModel.experiment_id == experiment.id)
            )
            trials = [t for t in trial_result.scalars().all() if t.value is not None]
            if not trials:
                return None

            direction = (experiment.objective or "minimize").lower()
            if direction == "maximize":
                return max(trials, key=lambda t: t.value)
            return min(trials, key=lambda t: t.value)

    async def get_experiment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验统计信息"""
        trials = await self.get_experiment_trials(experiment_id)
        async with self.session_factory() as db:
            exp_result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.id == uuid_lib.UUID(experiment_id))
            )
            experiment = exp_result.scalar_one_or_none()
        direction = (experiment.objective if experiment else "minimize") or "minimize"
        completed = [t for t in trials if (t.state or "").lower() in {"complete", "completed"} and t.value is not None]
        running = [t for t in trials if (t.state or "").lower() == "running"]
        failed = [t for t in trials if (t.state or "").lower() in {"fail", "failed"}]
        if direction.lower() == "maximize":
            best_value = max((t.value for t in completed), default=None)
        else:
            best_value = min((t.value for t in completed), default=None)
        average_value = (
            sum(t.value for t in completed) / len(completed) if completed else None
        )
        return {
            "total_trials": len(trials),
            "completed_trials": len(completed),
            "running_trials": len(running),
            "failed_trials": len(failed),
            "best_value": best_value,
            "average_value": average_value,
        }

    async def prune_trial(self, trial_id: str, reason: str) -> TrialModel | None:
        """剪枝试验"""
        return await self.update_trial_result(trial_id, None, "PRUNED", {"reason": reason})

    async def get_trial_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """获取试验历史"""
        trials = await self.get_experiment_trials(experiment_id)
        return [
            {
                "trial_id": str(trial.id),
                "value": trial.value,
                "parameters": trial.parameters,
                "timestamp": trial.end_time.isoformat() if trial.end_time else (trial.start_time.isoformat() if trial.start_time else utc_now().isoformat()),
            }
            for trial in trials
        ]

    async def search_experiments(self, query: str) -> List[ExperimentModel]:
        """搜索实验"""
        async with self.session_factory() as db:
            result = await db.execute(
                select(ExperimentModel).where(ExperimentModel.name.contains(query))
            )
            return result.scalars().all()
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        return self.resource_manager.get_resource_stats()
    
    def get_active_experiments(self) -> Dict[str, Any]:
        """获取活跃实验列表"""
        return {
            exp_id: {
                "experiment_name": exp_data["experiment"].name,
                "start_time": exp_data["start_time"].isoformat(),
                "status": exp_data["experiment"].status
            }
            for exp_id, exp_data in self.active_experiments.items()
        }
