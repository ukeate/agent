"""
超参数优化API接口

提供RESTful API用于管理超参数优化实验，包括创建、启动、监控、停止实验等功能。
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
from uuid import UUID
from src.ai.hyperparameter_optimization.experiment_manager import ExperimentManager
from src.ai.hyperparameter_optimization.search_engine import HyperparameterSearchEngine
from src.ai.hyperparameter_optimization.models import (
    ExperimentRequest,
    ExperimentResponse,
    ExperimentDetail,
    TrialResponse,
    OptimizationResult,
    AlgorithmComparison,
    ResourceStats,
    TaskInfo,
    CustomTaskRequest,
    OptimizationProgress,
    HyperparameterRangeSchema
)

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/hyperparameter-optimization", tags=["Hyperparameter Optimization"])

# 全局实例
experiment_manager = None
search_engine = None

def get_experiment_manager() -> ExperimentManager:
    """获取实验管理器实例"""
    global experiment_manager
    if experiment_manager is None:
        experiment_manager = ExperimentManager()
    return experiment_manager

def get_search_engine() -> HyperparameterSearchEngine:
    """获取搜索引擎实例"""
    global search_engine
    if search_engine is None:
        search_engine = HyperparameterSearchEngine()
    return search_engine

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """创建超参数优化实验"""
    try:
        return await manager.create_experiment(request)
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取所有实验列表"""
    try:
        return await manager.list_experiments()
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取特定实验的详细信息"""
    try:
        return await manager.get_experiment(str(experiment_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: UUID,
    background_tasks: BackgroundTasks,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """启动超参数优化实验"""
    try:
        await manager.start_experiment(str(experiment_id))
        return {"status": "started", "experiment_id": str(experiment_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """停止正在运行的实验"""
    try:
        return await manager.stop_experiment(str(experiment_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """删除实验及其所有数据"""
    try:
        return await manager.delete_experiment(str(experiment_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/trials", response_model=List[TrialResponse])
async def get_experiment_trials(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取实验的所有试验记录"""
    try:
        return await manager.get_trials(str(experiment_id))
    except Exception as e:
        logger.error(f"Failed to get trials for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/visualizations")
async def get_experiment_visualizations(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取实验的可视化图表"""
    try:
        return await manager.get_visualizations(str(experiment_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get visualizations for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/progress")
async def get_experiment_progress(
    experiment_id: UUID,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取实验的实时进度"""
    try:
        experiment_id_str = str(experiment_id)
        experiment = await manager.get_experiment(experiment_id_str)
        
        # 计算进度信息
        progress = OptimizationProgress(
            experiment_id=experiment_id_str,
            current_trial=experiment.total_trials,
            total_trials=json.loads(experiment.config).get("n_trials", 100),
            best_value=experiment.best_value,
            best_params=experiment.best_params,
            elapsed_time=(utc_now() - experiment.started_at).total_seconds() if experiment.started_at else 0,
            estimated_remaining_time=None,  # 可以基于历史数据计算
            status=experiment.status
        )
        
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 搜索引擎相关接口

@router.get("/tasks", response_model=List[str])
async def get_preset_tasks(
    engine: HyperparameterSearchEngine = Depends(get_search_engine)
):
    """获取预设任务类型列表"""
    try:
        return engine.get_preset_tasks()
    except Exception as e:
        logger.error(f"Failed to get preset tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_name}", response_model=TaskInfo)
async def get_task_info(
    task_name: str,
    engine: HyperparameterSearchEngine = Depends(get_search_engine)
):
    """获取特定任务的配置信息"""
    try:
        return engine.get_task_info(task_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get task info for {task_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks", response_model=str)
async def create_custom_task(
    request: CustomTaskRequest,
    engine: HyperparameterSearchEngine = Depends(get_search_engine)
):
    """创建自定义任务配置"""
    try:
        parameters = [param.model_dump() for param in request.parameters]
        
        return engine.create_custom_task(
            task_name=request.task_name,
            parameters=parameters,
            algorithm=request.algorithm,
            pruning=request.pruning,
            direction=request.direction,
            n_trials=request.n_trials,
            early_stopping=request.early_stopping,
            patience=request.patience
        )
    except Exception as e:
        logger.error(f"Failed to create custom task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/{task_name}")
async def optimize_for_task(
    task_name: str,
    background_tasks: BackgroundTasks,
    custom_config: Optional[Dict[str, Any]] = None,
    engine: HyperparameterSearchEngine = Depends(get_search_engine)
):
    """为指定任务执行优化（使用默认目标函数）"""
    try:
        # 这里需要提供默认的目标函数
        def default_objective(params: Dict[str, Any]) -> float:
            """默认演示目标函数"""
            total = 0
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    total += value ** 2
            return -total
        
        # 启动后台优化任务
        async def run_optimization():
            try:
                result = engine.optimize_for_task(task_name, default_objective)
                logger.info(f"Optimization for task {task_name} completed")
                return result
            except Exception as e:
                logger.error(f"Optimization for task {task_name} failed: {e}")
                raise
        
        background_tasks.add_task(run_optimization)
        
        return {
            "status": "optimization_started",
            "task_name": task_name,
            "message": "Optimization running in background"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start optimization for task {task_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-algorithms/{task_name}")
async def compare_algorithms(
    task_name: str,
    algorithms: List[str] = Query(default=["tpe", "cmaes", "random"]),
    background_tasks: BackgroundTasks = None,
    engine: HyperparameterSearchEngine = Depends(get_search_engine)
):
    """比较不同算法在指定任务上的表现"""
    try:
        from src.ai.hyperparameter_optimization.optimizer import OptimizationAlgorithm
        
        # 转换算法名称
        algorithm_enums = []
        for alg in algorithms:
            try:
                algorithm_enums.append(OptimizationAlgorithm(alg))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid algorithm: {alg}")
        
        # 默认目标函数
        def comparison_objective(params: Dict[str, Any]) -> float:
            """用于算法比较的目标函数"""
            # Ackley函数的简化版本
            total = 0
            count = 0
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    total += value ** 2
                    count += 1
            
            if count > 0:
                result = -20 * (total / count) ** 0.5 + 20
                return result
            else:
                return 0.0
        
        # 异步执行算法比较
        async def run_comparison():
            try:
                result = engine.compare_algorithms(task_name, comparison_objective, algorithm_enums)
                logger.info(f"Algorithm comparison for task {task_name} completed")
                return result
            except Exception as e:
                logger.error(f"Algorithm comparison for task {task_name} failed: {e}")
                raise
        
        if background_tasks:
            background_tasks.add_task(run_comparison)
            return {
                "status": "comparison_started",
                "task_name": task_name,
                "algorithms": algorithms,
                "message": "Algorithm comparison running in background"
            }
        else:
            # 同步执行（仅用于演示）
            return await run_comparison()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare algorithms for task {task_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 系统状态接口

@router.get("/resource-status", response_model=ResourceStats)
async def get_resource_status(
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取系统资源使用状态"""
    try:
        return manager.get_resource_status()
    except Exception as e:
        logger.error(f"Failed to get resource status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active-experiments")
async def get_active_experiments(
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """获取当前活跃的实验列表"""
    try:
        return manager.get_active_experiments()
    except Exception as e:
        logger.error(f"Failed to get active experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查接口

@router.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        from sqlalchemy import text
        manager = get_experiment_manager()
        db = manager.get_db()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": utc_now(),
            "services": {
                "database": "connected",
                "experiment_manager": "running",
                "search_engine": "running"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# 配置信息接口

@router.get("/algorithms")
async def get_available_algorithms():
    """获取可用的优化算法列表"""
    from src.ai.hyperparameter_optimization.optimizer import OptimizationAlgorithm
    
    return {
        "algorithms": [alg.value for alg in OptimizationAlgorithm],
        "descriptions": {
            "tpe": "Tree-structured Parzen Estimator - 高效贝叶斯优化",
            "cmaes": "Covariance Matrix Adaptation Evolution Strategy - 进化策略",
            "random": "Random Search - 随机搜索",
            "grid": "Grid Search - 网格搜索",
            "nsga2": "NSGA-II - 多目标遗传算法"
        },
        "parameters": {
            "tpe": {
                "n_startup_trials": {"type": "int", "default": 10, "min": 1, "max": 200, "description": "随机试验数量"},
                "n_ei_candidates": {"type": "int", "default": 24, "min": 10, "max": 200, "description": "EI候选点数量"},
                "multivariate": {"type": "boolean", "default": True, "description": "是否启用多变量建模"},
                "group": {"type": "boolean", "default": True, "description": "是否启用分组建模"}
            },
            "cmaes": {
                "n_startup_trials": {"type": "int", "default": 10, "min": 1, "max": 200, "description": "启动随机试验数量"},
                "independent_sampler": {"type": "select", "default": "tpe", "options": ["tpe"], "description": "独立采样器"}
            },
            "random": {},
            "grid": {},
            "nsga2": {}
        }
    }

@router.get("/pruning-strategies")
async def get_pruning_strategies():
    """获取可用的剪枝策略列表"""
    from src.ai.hyperparameter_optimization.optimizer import PruningAlgorithm
    
    return {
        "pruning_strategies": [prune.value for prune in PruningAlgorithm],
        "descriptions": {
            "median": "Median Pruner - 基于中位数的剪枝",
            "hyperband": "Hyperband - 基于连续减半的剪枝",
            "successive_halving": "Successive Halving - 连续减半剪枝",
            "none": "No Pruning - 不使用剪枝"
        }
    }

@router.get("/parameter-types")
async def get_parameter_types():
    """获取支持的参数类型"""
    return {
        "parameter_types": ["float", "int", "categorical", "boolean"],
        "descriptions": {
            "float": "浮点数参数，支持对数尺度",
            "int": "整数参数，支持步长和对数尺度",
            "categorical": "分类参数，从预定义选项中选择",
            "boolean": "布尔参数，True或False"
        },
        "examples": {
            "float": {
                "name": "learning_rate",
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": True
            },
            "int": {
                "name": "batch_size",
                "type": "int",
                "low": 16,
                "high": 128,
                "step": 16
            },
            "categorical": {
                "name": "optimizer",
                "type": "categorical",
                "choices": ["adam", "sgd", "rmsprop"]
            },
            "boolean": {
                "name": "use_batch_norm",
                "type": "boolean"
            }
        }
    }
