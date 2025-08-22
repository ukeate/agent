"""
奖励函数设计和优化系统

该模块实现多维度奖励计算框架，支持：
- 基础奖励函数（线性、非线性、阈值）
- 复合奖励计算（加权、乘积、分段）
- 动态奖励调整（自适应、基于进度）
- 奖励塑形和稀疏奖励处理
"""

from .base import RewardFunction, RewardConfig, RewardMetrics
from .basic_rewards import LinearReward, StepReward, ThresholdReward, GaussianReward
from .composite_rewards import WeightedReward, ProductReward, PiecewiseReward
from .adaptive_rewards import AdaptiveReward, ProgressBasedReward, CurriculumReward
from .shaping import PotentialBasedShaping, DifferenceReward, CuriosityReward

__all__ = [
    'RewardFunction',
    'RewardConfig', 
    'RewardMetrics',
    'LinearReward',
    'StepReward',
    'ThresholdReward',
    'GaussianReward',
    'WeightedReward',
    'ProductReward',
    'PiecewiseReward',
    'AdaptiveReward',
    'ProgressBasedReward',
    'CurriculumReward',
    'PotentialBasedShaping',
    'DifferenceReward',
    'CuriosityReward'
]