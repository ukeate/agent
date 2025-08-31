"""
自动化超参数优化系统

该模块提供基于Optuna的自动化超参数优化功能，支持多种优化算法、
智能剪枝、实验管理和可视化分析。

主要特性：
- 多算法支持：TPE、CMA-ES、随机搜索、网格搜索等
- 智能剪枝：早停机制，减少无效训练
- 实验管理：完整的实验生命周期管理
- 资源管理：GPU/CPU资源分配和监控
- 可视化：实时优化进度和参数重要性分析
"""

from .optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    HyperparameterRange,
    OptimizationAlgorithm,
    PruningAlgorithm,
    EarlyStoppingCallback,
    ResourceManager
)

from .experiment_manager import (
    ExperimentManager,
    ExperimentRequest,
    ExperimentResponse
)

from .search_engine import HyperparameterSearchEngine

__all__ = [
    "HyperparameterOptimizer",
    "OptimizationConfig",
    "HyperparameterRange", 
    "OptimizationAlgorithm",
    "PruningAlgorithm",
    "EarlyStoppingCallback",
    "ResourceManager",
    "ExperimentManager",
    "ExperimentRequest",
    "ExperimentResponse",
    "HyperparameterSearchEngine"
]