"""
多臂老虎机算法包

实现各种多臂老虎机算法，包括UCB、Thompson Sampling、Epsilon-Greedy等。
"""

from .base import MultiArmedBandit
from .ucb import UCBBandit
from .thompson_sampling import ThompsonSamplingBandit
from .epsilon_greedy import EpsilonGreedyBandit
from .contextual import LinearContextualBandit

__all__ = [
    "MultiArmedBandit",
    "UCBBandit",
    "ThompsonSamplingBandit", 
    "EpsilonGreedyBandit",
    "LinearContextualBandit"
]
