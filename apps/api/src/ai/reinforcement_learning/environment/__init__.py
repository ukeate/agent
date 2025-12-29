"""
强化学习环境模块

提供状态空间、动作空间、环境模拟和奖励函数等环境相关组件。
"""

from .state_space import (
    StateSpace,
    StateSpaceType,
    StateFeature,
    DiscreteStateSpace,
    ContinuousStateSpace,
    HybridStateSpace,
    StateSpaceFactory
)
from .action_space import (
    ActionSpace,
    ActionSpaceType,
    ActionDimension,
    DiscreteActionSpace,
    ContinuousActionSpace,
    MultiDiscreteActionSpace,
    HybridActionSpace,
    ActionSpaceFactory
)
from .simulator import (
    BaseEnvironment,
    AgentEnvironmentSimulator,
    EnvironmentStatus,
    EnvironmentInfo,
    StepResult,
    RewardFunction as SimRewardFunction,
    TransitionFunction,
    IdentityTransition,
    LinearTransition,
    GridWorldTransition,
    SparseReward,
    DenseReward,
    ShapedReward
)
from .reward_function import (
    BaseRewardFunction,
    RewardType,
    RewardComponent,
    SparseRewardFunction,
    DenseRewardFunction,
    ShapedRewardFunction,
    PotentialBasedRewardFunction,
    CompositeRewardFunction,
    RewardFunctionFactory

# 状态空间相关

# 动作空间相关

# 环境模拟器相关

# 奖励函数相关

# 网格世界环境 - 暂时注释，需要重构
# from .grid_world import GridWorldEnvironment

)

__all__ = [
    # 状态空间
    "StateSpace",
    "StateSpaceType", 
    "StateFeature",
    "DiscreteStateSpace",
    "ContinuousStateSpace",
    "HybridStateSpace",
    "StateSpaceFactory",
    
    # 动作空间
    "ActionSpace",
    "ActionSpaceType",
    "ActionDimension",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "MultiDiscreteActionSpace",
    "HybridActionSpace",
    "ActionSpaceFactory",
    
    # 环境模拟器
    "BaseEnvironment",
    "AgentEnvironmentSimulator",
    "EnvironmentStatus",
    "EnvironmentInfo",
    "StepResult",
    "SimRewardFunction",
    "TransitionFunction",
    "IdentityTransition",
    "LinearTransition",
    "GridWorldTransition",
    "SparseReward",
    "DenseReward",
    "ShapedReward",
    
    # 奖励函数
    "BaseRewardFunction",
    "RewardType",
    "RewardComponent",
    "SparseRewardFunction",
    "DenseRewardFunction",
    "ShapedRewardFunction",
    "PotentialBasedRewardFunction",
    "CompositeRewardFunction",
    "RewardFunctionFactory",
    
    # 具体环境
    # "GridWorldEnvironment"  # 暂时注释，需要重构
]
