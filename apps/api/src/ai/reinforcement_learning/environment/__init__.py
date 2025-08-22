"""
强化学习环境模块

提供状态空间、动作空间、环境模拟和奖励函数等环境相关组件。
"""

# 状态空间相关
from .state_space import (
    StateSpace,
    StateSpaceType,
    StateFeature,
    DiscreteStateSpace,
    ContinuousStateSpace,
    HybridStateSpace,
    StateSpaceFactory
)

# 动作空间相关
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

# 环境模拟器相关
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

# 奖励函数相关
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
)

# 网格世界环境 - 暂时注释，需要重构
# from .grid_world import GridWorldEnvironment

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