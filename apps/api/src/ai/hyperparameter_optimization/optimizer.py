"""
超参数优化核心引擎

基于Optuna实现的多算法超参数优化系统，支持TPE、CMA-ES、随机搜索等算法，
提供智能剪枝、早停机制和资源管理功能。
"""

import optuna
import optuna.visualization as vis
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, GridSampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import psutil

from src.core.logging import get_logger
logger = get_logger(__name__)

class OptimizationAlgorithm(Enum):
    """优化算法枚举"""
    TPE = "tpe"
    CMAES = "cmaes" 
    RANDOM = "random"
    GRID = "grid"
    NSGA2 = "nsga2"

class PruningAlgorithm(Enum):
    """剪枝算法枚举"""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"

@dataclass
class HyperparameterRange:
    """超参数范围定义"""
    name: str
    type: str  # 'float', 'int', 'categorical', 'boolean'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None

@dataclass
class OptimizationConfig:
    """优化配置"""
    study_name: str
    algorithm: OptimizationAlgorithm
    pruning: PruningAlgorithm
    direction: str  # 'maximize', 'minimize'
    
    # 优化参数
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    
    # 早停参数
    early_stopping: bool = True
    patience: int = 20
    min_improvement: float = 0.001
    
    # 资源管理
    max_concurrent_trials: int = 5
    resource_timeout: int = 3600
    
    # 存储配置
    storage_url: Optional[str] = None
    load_if_exists: bool = True

class HyperparameterOptimizer:
    """超参数优化器核心引擎"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = None
        self.parameter_ranges = {}
        self.trial_history = []
        self.optimization_stats = {
            "start_time": None,
            "end_time": None,
            "best_value": None,
            "best_params": None,
            "total_trials": 0,
            "successful_trials": 0,
            "pruned_trials": 0
        }
        
        self.logger = get_logger(__name__)
        
        # 初始化研究（GRID需要参数范围后再创建）
        if self.config.algorithm != OptimizationAlgorithm.GRID:
            self._setup_study()
    
    def _setup_study(self):
        """设置Optuna研究"""
        
        # 选择采样器
        sampler = self._create_sampler()
        
        # 选择剪枝器
        pruner = self._create_pruner()
        
        # 创建研究
        study_kwargs = {
            'study_name': self.config.study_name,
            'direction': self.config.direction,
            'storage': self.config.storage_url,
            'load_if_exists': self.config.load_if_exists,
            'pruner': pruner
        }
        
        # 只有非None采样器才添加sampler参数
        if sampler is not None:
            study_kwargs['sampler'] = sampler
            
        self.study = optuna.create_study(**study_kwargs)
        
        self.logger.info(f"Created study: {self.config.study_name}")
        self.logger.info(f"Algorithm: {self.config.algorithm.value}")
        self.logger.info(f"Pruning: {self.config.pruning.value}")
    
    def _create_sampler(self):
        """创建采样器"""
        
        if self.config.algorithm == OptimizationAlgorithm.TPE:
            return TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            )
        elif self.config.algorithm == OptimizationAlgorithm.CMAES:
            return CmaEsSampler(
                n_startup_trials=10,
                independent_sampler=TPESampler()
            )
        elif self.config.algorithm == OptimizationAlgorithm.RANDOM:
            return RandomSampler()
        elif self.config.algorithm == OptimizationAlgorithm.GRID:
            search_space = self._build_grid_search_space()
            if not search_space:
                raise ValueError("GRID算法需要有效的参数搜索空间")
            return GridSampler(search_space)
        elif self.config.algorithm == OptimizationAlgorithm.NSGA2:
            return NSGAIISampler()
        else:
            return TPESampler()

    def _build_grid_search_space(self) -> Dict[str, List[Any]]:
        """构建网格搜索空间"""
        search_space: Dict[str, List[Any]] = {}
        for name, config in self.parameter_ranges.items():
            if isinstance(config, HyperparameterRange):
                param_type = config.type
                low = config.low
                high = config.high
                choices = config.choices
                step = config.step
            else:
                param_type = config.get("type")
                low = config.get("low")
                high = config.get("high")
                choices = config.get("choices")
                step = config.get("step")

            if param_type == "categorical":
                if not choices:
                    continue
                search_space[name] = list(choices)
            elif param_type == "int":
                if low is None or high is None:
                    continue
                step_value = int(step or 1)
                search_space[name] = list(range(int(low), int(high) + 1, step_value))
            elif param_type == "float":
                if low is None or high is None:
                    continue
                if step:
                    values = []
                    value = float(low)
                    while value <= float(high) + 1e-12:
                        values.append(float(value))
                        value += float(step)
                else:
                    values = [float(low), (float(low) + float(high)) / 2, float(high)]
                search_space[name] = values
        return search_space
    
    def _create_pruner(self):
        """创建剪枝器"""
        
        if self.config.pruning == PruningAlgorithm.MEDIAN:
            return MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        elif self.config.pruning == PruningAlgorithm.HYPERBAND:
            return HyperbandPruner(
                min_resource=10,
                max_resource=100,
                reduction_factor=3
            )
        elif self.config.pruning == PruningAlgorithm.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner(
                min_resource=10,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        else:
            return optuna.pruners.NopPruner()
    
    def add_parameter_range(self, param_range: HyperparameterRange):
        """添加参数搜索范围"""
        self.parameter_ranges[param_range.name] = param_range
        self.logger.info(f"Added parameter: {param_range.name} ({param_range.type})")
        if self.config.algorithm == OptimizationAlgorithm.GRID and self.study is None:
            self._setup_study()
    
    def add_parameter_ranges(self, ranges: Dict[str, Dict[str, Any]]):
        """添加多个参数搜索范围"""
        self.parameter_ranges = ranges
        if self.config.algorithm == OptimizationAlgorithm.GRID and self.study is None:
            self._setup_study()
    
    def suggest_parameters(self, trial) -> Dict[str, Any]:
        """建议参数（兼容方法）"""
        return self.suggest_hyperparameters(trial)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议超参数"""
        
        suggested_params = {}
        
        for name, config in self.parameter_ranges.items():
            # 处理字典形式的参数配置
            if isinstance(config, dict):
                param_type = config.get('type')
                if param_type == 'float':
                    log_scale = config.get('log', False)
                    if log_scale:
                        suggested_params[name] = trial.suggest_float(
                            name, 
                            config['low'], 
                            config['high'],
                            log=True
                        )
                    else:
                        suggested_params[name] = trial.suggest_float(
                            name, 
                            config['low'], 
                            config['high']
                        )
                elif param_type == 'int':
                    suggested_params[name] = trial.suggest_int(
                        name,
                        int(config['low']),
                        int(config['high']),
                        step=int(config.get('step', 1))
                    )
                elif param_type == 'categorical':
                    suggested_params[name] = trial.suggest_categorical(
                        name,
                        config['choices']
                    )
                else:
                    raise ValueError(f"不支持的参数类型: {param_type}")
            # 处理HyperparameterRange对象
            elif hasattr(config, 'type'):
                param_range = config
                if param_range.type == 'float':
                    if hasattr(param_range, 'log') and param_range.log:
                        suggested_params[name] = trial.suggest_float(
                            name, 
                            param_range.low, 
                            param_range.high,
                            log=True
                        )
                    else:
                        suggested_params[name] = trial.suggest_float(
                            name, 
                            param_range.low, 
                            param_range.high
                        )
                elif param_range.type == 'int':
                    suggested_params[name] = trial.suggest_int(
                        name,
                        int(param_range.low),
                        int(param_range.high),
                        step=int(param_range.step) if param_range.step else 1
                    )
                elif param_range.type == 'categorical':
                    suggested_params[name] = trial.suggest_categorical(
                        name,
                        param_range.choices
                    )
                elif param_range.type == 'boolean':
                    suggested_params[name] = trial.suggest_categorical(
                        name,
                        [True, False]
                    )
        
        return suggested_params
    
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, Any]], float],
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """执行优化"""
        if self.study is None:
            self._setup_study()
        self.optimization_stats["start_time"] = utc_now()
        
        def wrapped_objective(trial):
            # 获取建议参数
            params = self.suggest_parameters(trial)
            
            # 执行目标函数
            try:
                value = objective_function(params)
                
                # 记录试验历史
                self.trial_history.append({
                    "trial_number": trial.number,
                    "params": params,
                    "value": value,
                    "state": "COMPLETE",
                    "timestamp": utc_now().isoformat()
                })
                
                return value
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {e}")
                
                # 记录失败试验
                self.trial_history.append({
                    "trial_number": trial.number,
                    "params": params,
                    "value": None,
                    "state": "FAIL",
                    "error": str(e),
                    "timestamp": utc_now().isoformat()
                })
                
                raise optuna.TrialPruned()
        
        # 添加回调函数
        study_callbacks = []
        if callbacks:
            study_callbacks.extend(callbacks)
        
        # 添加早停回调
        if self.config.early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                patience=self.config.patience,
                min_improvement=self.config.min_improvement
            )
            study_callbacks.append(early_stopping_callback)
        
        self.logger.info(f"Starting optimization with {self.config.n_trials} trials")
        
        # 执行优化
        self.study.optimize(
            wrapped_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=study_callbacks,
            show_progress_bar=True
        )
        
        # 更新统计
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:
            best_value = self.study.best_value
            best_params = self.study.best_params
        else:
            best_value = None
            best_params = {}
            
        self.optimization_stats.update({
            "end_time": utc_now(),
            "best_value": best_value,
            "best_params": best_params,
            "total_trials": len(self.study.trials),
            "successful_trials": len(completed_trials),
            "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        })
        
        self.logger.info(f"Optimization completed")
        self.logger.info(f"Best value: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
            "stats": self.optimization_stats,
            "study": self.study
        }
    
    def should_prune_trial(self, trial, intermediate_value: float, step: int) -> bool:
        """判断是否应该剪枝试验"""
        trial.report(intermediate_value, step)
        return trial.should_prune()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史（返回列表）"""
        return [
            {
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name if hasattr(trial.state, 'name') else str(trial.state)
            }
            for trial in self.study.trials
        ]
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """获取参数重要性"""
        try:
            import optuna.importance
            return optuna.importance.get_param_importances(self.study)
        except:
            return {}
    
    async def optimize_async(self, objective_function) -> Dict[str, Any]:
        """异步优化"""
        if not asyncio.iscoroutinefunction(objective_function):
            return await asyncio.to_thread(self.optimize, objective_function)

        def wrapped_objective(trial):
            params = self.suggest_parameters(trial)
            return asyncio.run(objective_function(params))

        return await asyncio.to_thread(self.optimize, wrapped_objective)
    
    def create_visualizations(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """创建可视化图表"""
        
        if len(self.study.trials) < 2:
            self.logger.warning("Not enough trials for visualization")
            return {}
        
        visualizations = {}
        
        try:
            # 优化历史图
            fig_history = vis.plot_optimization_history(self.study)
            visualizations["optimization_history"] = fig_history
            
            # 参数重要性图
            if len(self.study.trials) > 10:
                fig_importance = vis.plot_param_importances(self.study)
                visualizations["param_importance"] = fig_importance
            
            # 参数关系图
            if len(self.parameter_ranges) > 1:
                fig_parallel = vis.plot_parallel_coordinate(self.study)
                visualizations["parallel_coordinate"] = fig_parallel
            
            # 参数分布图
            fig_slice = vis.plot_slice(self.study)
            visualizations["param_slice"] = fig_slice
            
            # 等高线图
            if len(self.parameter_ranges) >= 2:
                param_names = list(self.parameter_ranges.keys())[:2]
                fig_contour = vis.plot_contour(
                    self.study,
                    params=param_names
                )
                visualizations["contour"] = fig_contour
            
            # 保存图表
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                
                for name, fig in visualizations.items():
                    fig.write_html(f"{save_path}/{name}.html")
                    fig.write_image(f"{save_path}/{name}.png")
            
            self.logger.info(f"Created {len(visualizations)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return visualizations
    
    def save_study(self, file_path: str):
        """保存研究到文件"""
        
        study_data = {
            "config": asdict(self.config),
            "parameter_ranges": {name: asdict(param) for name, param in self.parameter_ranges.items()},
            "optimization_stats": self.optimization_stats,
            "trial_history": self.trial_history
        }
        
        with open(file_path, 'w') as f:
            json.dump(study_data, f, indent=2, default=str)
        
        self.logger.info(f"Study saved to {file_path}")
    
    def load_study(self, file_path: str):
        """从文件加载研究"""
        
        with open(file_path, 'r') as f:
            study_data = json.load(f)
        
        # 恢复参数范围
        for name, param_data in study_data["parameter_ranges"].items():
            param_range = HyperparameterRange(**param_data)
            self.parameter_ranges[name] = param_range
        
        # 恢复统计数据
        self.optimization_stats = study_data["optimization_stats"]
        self.trial_history = study_data["trial_history"]
        
        self.logger.info(f"Study loaded from {file_path}")

class EarlyStoppingCallback:
    """早停回调函数"""
    
    def __init__(self, patience: int = 20, min_improvement: float = 0.001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.patience_counter = 0
        
        self.logger = get_logger(__name__)
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """执行早停检查"""
        
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        current_value = trial.value
        
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # 检查是否有改进
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            improvement = current_value - self.best_value
            if current_value > self.best_value:
                self.best_value = current_value
        else:
            improvement = self.best_value - current_value
            if current_value < self.best_value:
                self.best_value = current_value
        
        if improvement >= self.min_improvement:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.logger.info(
            f"Trial {trial.number}: value={current_value:.6f}, "
            f"improvement={improvement:.6f}, patience={self.patience_counter}/{self.patience}"
        )
        
        # 检查是否需要早停
        if self.patience_counter >= self.patience:
            self.logger.info(f"Early stopping triggered after {trial.number} trials")
            study.stop()

class ResourceManager:
    """资源管理器"""
    
    def __init__(self, max_concurrent: int = 5, max_gpu_memory: float = 0.9):
        self.max_concurrent = max_concurrent
        self.max_gpu_memory = max_gpu_memory
        self.current_trials = {}
        self.resource_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "gpu_memory": 0.0
        }
        
        self.logger = get_logger(__name__)
        self._monitor_resources()
    
    def _monitor_resources(self):
        """监控资源使用"""
        
        # CPU和内存监控
        self.resource_usage["cpu"] = psutil.cpu_percent()
        self.resource_usage["memory"] = psutil.virtual_memory().percent
        
        # GPU监控 (如果有NVIDIA GPU)
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.resource_usage["gpu_memory"] = mem_info.used / max(mem_info.total, 1)
        except Exception:
            self.logger.debug("GPU资源监控不可用", exc_info=True)
            self.resource_usage["gpu_memory"] = 0.0
    
    def can_start_trial(self) -> bool:
        """检查是否可以启动新试验"""
        
        # 检查并发数限制
        if len(self.current_trials) >= self.max_concurrent:
            return False
        
        # 检查GPU内存使用
        if self.resource_usage["gpu_memory"] > self.max_gpu_memory:
            return False
        
        return True
    
    def register_trial(self, trial_id: str, resource_requirement: Dict[str, float]):
        """注册试验资源需求"""
        
        self.current_trials[trial_id] = {
            "start_time": utc_now(),
            "resource_requirement": resource_requirement
        }
        
        self.logger.info(f"Trial {trial_id} registered with resources: {resource_requirement}")
    
    def unregister_trial(self, trial_id: str):
        """注销试验"""
        
        if trial_id in self.current_trials:
            trial_info = self.current_trials.pop(trial_id)
            duration = utc_now() - trial_info["start_time"]
            
            self.logger.info(f"Trial {trial_id} completed in {duration}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        
        return {
            "current_trials": len(self.current_trials),
            "max_concurrent": self.max_concurrent,
            "resource_usage": self.resource_usage,
            "active_trials": list(self.current_trials.keys())
        }
