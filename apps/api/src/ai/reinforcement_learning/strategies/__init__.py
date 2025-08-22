"""
策略优化和训练管理模块

该模块实现各种探索策略和训练管理功能：
- 探索策略（epsilon-greedy, UCB, Thompson sampling等）
- 训练调度和学习率管理
- 策略评估和选择
- 超参数优化
"""

from .exploration import (
    ExplorationStrategy,
    EpsilonGreedyStrategy,
    DecayingEpsilonGreedyStrategy,
    UCBStrategy,
    ThompsonSamplingStrategy,
    BoltzmannExplorationStrategy,
    NoiseBasedExplorationStrategy,
    CuriosityDrivenExploration
)

from .training_manager import (
    TrainingManager,
    TrainingConfig,
    TrainingMetrics,
    LearningRateScheduler,
    EarlyStopping,
    PerformanceTracker
)

from .policy_evaluation import (
    PolicyEvaluator,
    PolicyComparison,
    PerformanceMetrics,
    StatisticalTesting
)

from .hyperparameter_optimization import (
    HyperparameterOptimizer,
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    PopulationBasedTraining
)

__all__ = [
    # 探索策略
    'ExplorationStrategy',
    'EpsilonGreedyStrategy', 
    'DecayingEpsilonGreedyStrategy',
    'UCBStrategy',
    'ThompsonSamplingStrategy',
    'BoltzmannExplorationStrategy',
    'NoiseBasedExplorationStrategy',
    'CuriosityDrivenExploration',
    
    # 训练管理
    'TrainingManager',
    'TrainingConfig',
    'TrainingMetrics',
    'LearningRateScheduler',
    'EarlyStopping',
    'PerformanceTracker',
    
    # 策略评估
    'PolicyEvaluator',
    'PolicyComparison',
    'PerformanceMetrics',
    'StatisticalTesting',
    
    # 超参数优化
    'HyperparameterOptimizer',
    'GridSearch',
    'RandomSearch',
    'BayesianOptimization',
    'PopulationBasedTraining'
]