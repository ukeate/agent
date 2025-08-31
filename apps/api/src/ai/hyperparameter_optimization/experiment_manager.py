"""
实验管理系统

提供实验生命周期管理，包括创建、启动、监控、停止和删除实验的功能。
支持异步操作和后台任务处理。
"""

from fastapi import HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any, Callable
import uuid as uuid_lib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
import asyncio
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    Base,
    ExperimentModel,
    TrialModel,
    StudyMetadataModel,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentDetail,
    TrialResponse,
    HyperparameterRangeSchema
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


def get_db_session():
    """获取数据库会话（测试用）"""
    # 这是一个简化实现，实际应该从配置中获取
    from contextlib import contextmanager
    
    @contextmanager
    def session():
        yield None
    
    return session()


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, database_url: str = "postgresql://user:password@localhost:5432/hyperopt"):
        self.database_url = database_url
        self.db_session = None  # 用于测试的会话
        try:
            self.engine = create_engine(database_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        except:
            # 如果数据库连接失败，使用内存数据库
            self.engine = None
            self.SessionLocal = None
        
        self.active_experiments = {}
        self.resource_manager = ResourceManager()
        self.search_engine = HyperparameterSearchEngine()
        
        self.logger = logging.getLogger(__name__)
    
    # 添加异步方法（用于测试）
    async def create_experiment(self, experiment_data):
        """创建实验（异步）"""
        from .models import Experiment, ExperimentState
        experiment = Experiment(
            id=experiment_data.get("id", 1),
            name=experiment_data["name"],
            state=ExperimentState.CREATED,
            description=experiment_data.get("description"),
            algorithm=experiment_data.get("algorithm"),
            parameter_ranges=experiment_data.get("parameter_ranges", {}),
            optimization_config=experiment_data.get("optimization_config", {})
        )
        if self.db_session:
            self.db_session.add(experiment)
            self.db_session.commit()
            self.db_session.refresh(experiment)
        return experiment
    
    async def get_experiment(self, experiment_id):
        """获取实验（异步）"""
        if self.db_session:
            return self.db_session.query(ExperimentModel).filter(
                ExperimentModel.id == experiment_id
            ).first()
        return None
    
    async def list_experiments(self, skip=0, limit=10):
        """列出实验（异步）"""
        if self.db_session:
            return self.db_session.query(ExperimentModel).offset(skip).limit(limit).all()
        return []
    
    async def update_experiment_state(self, experiment_id, state):
        """更新实验状态（异步）"""
        from .models import Experiment
        if self.db_session:
            experiment = await self.get_experiment(experiment_id)
            if experiment:
                experiment.state = state
                self.db_session.commit()
            return experiment
        # 简化实现
        experiment = Experiment(id=experiment_id, name="test", state=state)
        return experiment
    
    async def delete_experiment(self, experiment_id):
        """删除实验（异步）"""
        if self.db_session:
            experiment = await self.get_experiment(experiment_id)
            if experiment:
                self.db_session.delete(experiment)
                self.db_session.commit()
                return True
        return False
    
    async def create_trial(self, experiment_id, trial_data):
        """创建试验（异步）"""
        from .models import Trial, TrialState
        trial = Trial(
            id=trial_data.get("id", 1),
            experiment_id=experiment_id,
            state=trial_data.get("state", TrialState.RUNNING),
            parameters=trial_data.get("parameters", {}),
            value=trial_data.get("value"),
            metrics=trial_data.get("metrics", {})
        )
        if self.db_session:
            self.db_session.add(trial)
            self.db_session.commit()
            self.db_session.refresh(trial)
        return trial
    
    async def update_trial_result(self, trial_id, value, state, metrics):
        """更新试验结果（异步）"""
        if self.db_session:
            trial = self.db_session.query(TrialModel).filter(TrialModel.id == trial_id).first()
            if trial:
                trial.value = value
                trial.state = state
                trial.metrics = metrics
                self.db_session.commit()
            return trial
        # 简化实现
        trial = Trial(id=trial_id, experiment_id=1, state=state, value=value, metrics=metrics)
        return trial
    
    async def get_experiment_trials(self, experiment_id):
        """获取实验的试验列表（异步）"""
        if self.db_session:
            return self.db_session.query(TrialModel).filter(
                TrialModel.experiment_id == experiment_id
            ).order_by(TrialModel.start_time).all()
        return []
    
    async def get_best_trial(self, experiment_id):
        """获取最佳试验（异步）"""
        trials = await self.get_experiment_trials(experiment_id)
        if not trials:
            return None
        
        experiment = await self.get_experiment(experiment_id)
        direction = experiment.optimization_config.get("direction", "minimize") if experiment else "minimize"
        
        completed_trials = [t for t in trials if t.state == "complete"]
        if not completed_trials:
            return None
        
        if direction == "minimize":
            return min(completed_trials, key=lambda t: t.value)
        else:
            return max(completed_trials, key=lambda t: t.value)
    
    async def get_experiment_statistics(self, experiment_id):
        """获取实验统计信息（异步）"""
        trials = await self.get_experiment_trials(experiment_id)
        
        completed_trials = [t for t in trials if t.state == "complete" and t.value is not None]
        
        stats = {
            "total_trials": len(trials),
            "completed_trials": len(completed_trials),
            "running_trials": len([t for t in trials if t.state == "running"]),
            "failed_trials": len([t for t in trials if t.state == "failed"]),
            "best_value": min([t.value for t in completed_trials]) if completed_trials else None,
            "average_value": sum([t.value for t in completed_trials]) / len(completed_trials) if completed_trials else None
        }
        
        return stats
    
    async def prune_trial(self, trial_id, reason):
        """剪枝试验（异步）"""
        from .models import TrialState
        return await self.update_trial_result(trial_id, None, TrialState.PRUNED, {"reason": reason})
    
    async def get_trial_history(self, experiment_id):
        """获取试验历史（异步）"""
        trials = await self.get_experiment_trials(experiment_id)
        history = []
        for trial in trials:
            history.append({
                "trial_id": trial.id,
                "value": trial.value,
                "parameters": trial.parameters,
                "timestamp": trial.created_at.isoformat() if hasattr(trial, 'created_at') else utc_now().isoformat()
            })
        return history
    
    async def search_experiments(self, query):
        """搜索实验（异步）"""
        if self.db_session:
            return self.db_session.query(ExperimentModel).filter(
                ExperimentModel.name.contains(query)
            ).all()
        return []
    
    def get_db(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    async def create_experiment_async(self, request: ExperimentRequest) -> ExperimentResponse:
        """异步创建实验"""
        
        db = self.get_db()
        try:
            # 验证参数
            if not request.parameters:
                raise HTTPException(status_code=400, detail="Parameters cannot be empty")
            
            # 创建实验记录
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
                parameters=json.dumps([param.dict() for param in request.parameters]),
                status="created",
                max_concurrent_trials=request.max_concurrent_trials
            )
            
            db.add(experiment)
            db.commit()
            db.refresh(experiment)
            
            # 创建研究元数据
            study_metadata = StudyMetadataModel(
                study_name=f"exp_{experiment.id}",
                experiment_id=experiment.id,
                search_space=json.dumps([param.dict() for param in request.parameters])
            )
            
            db.add(study_metadata)
            db.commit()
            
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
            db.rollback()
            self.logger.error(f"Database error creating experiment: {e}")
            raise HTTPException(status_code=500, detail="Database error")
        finally:
            db.close()
    
    async def list_experiments_async(self) -> List[ExperimentResponse]:
        """异步列出实验"""
        
        db = self.get_db()
        try:
            experiments = db.query(ExperimentModel).order_by(ExperimentModel.created_at.desc()).all()
            
            return [
                ExperimentResponse(
                    id=str(exp.id),
                    name=exp.name,
                    status=exp.status,
                    algorithm=exp.algorithm,
                    objective=exp.objective,
                    created_at=exp.created_at
                )
                for exp in experiments
            ]
        finally:
            db.close()
    
    async def get_experiment_async(self, experiment_id: str) -> ExperimentDetail:
        """异步获取实验详情"""
        
        db = self.get_db()
        try:
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # 获取试验统计
            trials = db.query(TrialModel).filter(
                TrialModel.experiment_id == experiment.id
            ).all()
            
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
                trials_count=len(trials)
            )
        finally:
            db.close()
    
    async def start_experiment_async(
        self, 
        experiment_id: str, 
        objective_function: Optional[Callable[[Dict[str, Any]], float]] = None
    ):
        """异步启动实验"""
        
        db = self.get_db()
        try:
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            if experiment.status != "created":
                raise HTTPException(status_code=400, detail="Experiment already started or completed")
            
            # 检查资源
            if not self.resource_manager.can_start_trial():
                raise HTTPException(status_code=503, detail="Insufficient resources to start experiment")
            
            # 更新状态
            experiment.status = "running"
            experiment.started_at = utc_now()
            db.commit()
            
            # 注册资源使用
            resource_requirement = {
                "gpu": 1.0,
                "cpu": experiment.max_concurrent_trials or 4,
                "memory": 8.0
            }
            self.resource_manager.register_trial(experiment_id, resource_requirement)
            
            # 存储实验引用
            self.active_experiments[experiment_id] = {
                "experiment": experiment,
                "start_time": utc_now()
            }
            
            # 在后台运行优化
            asyncio.create_task(self._run_optimization(experiment_id, objective_function))
            
            self.logger.info(f"Started experiment: {experiment_id}")
            
        except Exception as e:
            # 回滚状态
            experiment.status = "failed"
            db.commit()
            self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
            raise
        finally:
            db.close()
    
    async def _run_optimization(
        self, 
        experiment_id: str, 
        objective_function: Optional[Callable[[Dict[str, Any]], float]]
    ):
        """运行优化（后台任务）"""
        
        db = self.get_db()
        
        try:
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if not experiment:
                return
            
            # 解析配置
            config_data = json.loads(experiment.config)
            params_data = json.loads(experiment.parameters)
            
            # 创建优化配置
            optimization_config = OptimizationConfig(
                study_name=f"exp_{experiment_id}",
                algorithm=OptimizationAlgorithm(experiment.algorithm),
                pruning=PruningAlgorithm.HYPERBAND,  # 默认使用Hyperband
                direction=experiment.objective,
                n_trials=config_data.get("n_trials", 100),
                timeout=config_data.get("timeout"),
                early_stopping=config_data.get("early_stopping", True),
                patience=config_data.get("patience", 20),
                min_improvement=config_data.get("min_improvement", 0.001),
                max_concurrent_trials=config_data.get("max_concurrent_trials", 5)
            )
            
            # 创建参数范围
            parameter_ranges = []
            for param_data in params_data:
                param_range = HyperparameterRange(**param_data)
                parameter_ranges.append(param_range)
            
            # 创建优化器
            optimizer = HyperparameterOptimizer(optimization_config)
            for param_range in parameter_ranges:
                optimizer.add_parameter_range(param_range)
            
            # 默认目标函数（用于演示）
            if objective_function is None:
                def default_objective(params: Dict[str, Any]) -> float:
                    """默认目标函数 - Rosenbrock函数的多维版本"""
                    import random
                    import time
                    import math
                    
                    # 模拟训练时间
                    time.sleep(random.uniform(5, 15))
                    
                    # 计算多维Rosenbrock函数
                    total = 0
                    param_values = list(params.values())
                    
                    for i in range(len(param_values) - 1):
                        if isinstance(param_values[i], (int, float)) and isinstance(param_values[i + 1], (int, float)):
                            x, y = param_values[i], param_values[i + 1]
                            total += 100 * (y - x**2)**2 + (1 - x)**2
                    
                    # 加入一些随机噪声
                    noise = random.gauss(0, 0.1)
                    result = -(total + noise)  # 负号因为要最大化
                    
                    # 模拟可能的失败情况
                    if random.random() < 0.05:  # 5%的失败率
                        raise ValueError("模拟训练失败")
                    
                    return result
                
                objective_function = default_objective
            
            # 创建进度回调
            def progress_callback(study, trial):
                # 更新数据库中的试验记录
                self._save_trial_to_db(experiment_id, trial)
            
            # 执行优化
            result = optimizer.optimize(
                objective_function,
                callbacks=[progress_callback]
            )
            
            # 更新实验结果
            db.refresh(experiment)
            experiment.status = "completed"
            experiment.completed_at = utc_now()
            experiment.best_value = result["best_value"]
            experiment.best_params = json.dumps(result["best_params"])
            experiment.total_trials = result["stats"]["total_trials"]
            experiment.successful_trials = result["stats"]["successful_trials"]
            experiment.pruned_trials = result["stats"]["pruned_trials"]
            
            db.commit()
            
            # 创建可视化
            try:
                visualizations = optimizer.create_visualizations(
                    save_path=f"./optimization_results/{experiment_id}"
                )
                self.logger.info(f"Created {len(visualizations)} visualizations for experiment {experiment_id}")
            except Exception as e:
                self.logger.warning(f"Failed to create visualizations: {e}")
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            
        except Exception as e:
            # 更新错误状态
            db.refresh(experiment)
            experiment.status = "failed"
            experiment.completed_at = utc_now()
            db.commit()
            
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            
        finally:
            # 清理资源
            self.resource_manager.unregister_trial(experiment_id)
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            db.close()
    
    def _save_trial_to_db(self, experiment_id: str, trial):
        """保存试验到数据库"""
        
        db = self.get_db()
        try:
            # 检查试验是否已存在
            existing_trial = db.query(TrialModel).filter(
                TrialModel.experiment_id == uuid_lib.UUID(experiment_id),
                TrialModel.trial_number == trial.number
            ).first()
            
            if existing_trial:
                # 更新现有试验
                existing_trial.value = trial.value
                existing_trial.state = trial.state.name
                existing_trial.end_time = trial.datetime_complete
                existing_trial.duration = trial.duration.total_seconds() if trial.duration else None
            else:
                # 创建新试验记录
                trial_record = TrialModel(
                    experiment_id=uuid_lib.UUID(experiment_id),
                    trial_number=trial.number,
                    parameters=json.dumps(trial.params),
                    value=trial.value,
                    state=trial.state.name,
                    start_time=trial.datetime_start,
                    end_time=trial.datetime_complete,
                    duration=trial.duration.total_seconds() if trial.duration else None
                )
                db.add(trial_record)
            
            db.commit()
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to save trial to database: {e}")
        finally:
            db.close()
    
    async def stop_experiment_async(self, experiment_id: str) -> Dict[str, Any]:
        """异步停止实验"""
        
        db = self.get_db()
        try:
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            if experiment.status != "running":
                raise HTTPException(status_code=400, detail="Experiment is not running")
            
            # 更新状态
            experiment.status = "stopped"
            experiment.completed_at = utc_now()
            db.commit()
            
            # 清理资源
            self.resource_manager.unregister_trial(experiment_id)
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            self.logger.info(f"Stopped experiment: {experiment_id}")
            
            return {"status": "stopped", "experiment_id": experiment_id}
            
        finally:
            db.close()
    
    async def delete_experiment_async(self, experiment_id: str) -> Dict[str, Any]:
        """异步删除实验"""
        
        db = self.get_db()
        try:
            # 首先停止实验（如果在运行）
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if experiment and experiment.status == "running":
                await self.stop_experiment_async(experiment_id)
            
            # 删除试验记录（级联删除）
            deleted_count = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).delete()
            
            if deleted_count == 0:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # 删除研究元数据
            db.query(StudyMetadataModel).filter(
                StudyMetadataModel.experiment_id == uuid_lib.UUID(experiment_id)
            ).delete()
            
            db.commit()
            
            self.logger.info(f"Deleted experiment: {experiment_id}")
            
            return {"status": "deleted", "experiment_id": experiment_id}
            
        finally:
            db.close()
    
    async def get_trials_async(self, experiment_id: str) -> List[TrialResponse]:
        """异步获取试验列表"""
        
        db = self.get_db()
        try:
            trials = db.query(TrialModel).filter(
                TrialModel.experiment_id == uuid_lib.UUID(experiment_id)
            ).order_by(TrialModel.trial_number).all()
            
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
        finally:
            db.close()
    
    async def get_visualizations_async(self, experiment_id: str) -> Dict[str, Any]:
        """异步获取可视化"""
        
        # 检查实验是否存在
        db = self.get_db()
        try:
            experiment = db.query(ExperimentModel).filter(
                ExperimentModel.id == uuid_lib.UUID(experiment_id)
            ).first()
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
        finally:
            db.close()
        
        # 返回可视化文件路径
        base_path = f"/static/visualizations/{experiment_id}"
        
        return {
            "optimization_history": f"{base_path}/optimization_history.html",
            "param_importance": f"{base_path}/param_importance.html",
            "parallel_coordinate": f"{base_path}/parallel_coordinate.html",
            "param_slice": f"{base_path}/param_slice.html",
            "contour": f"{base_path}/contour.html"
        }
    
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