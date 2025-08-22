"""
动作空间定义和管理框架

定义和管理强化学习环境中的动作空间，支持离散、连续和混合动作空间。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json


class ActionSpaceType(Enum):
    """动作空间类型"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"
    HYBRID = "hybrid"


@dataclass
class ActionDimension:
    """动作维度定义"""
    name: str
    action_type: str  # "discrete", "continuous", "categorical"
    low: Optional[float] = None  # 最小值（连续动作）
    high: Optional[float] = None  # 最大值（连续动作）
    n: Optional[int] = None  # 离散动作数量
    categories: Optional[List[str]] = None  # 动作类别
    description: Optional[str] = None
    
    def validate_action(self, action: Any) -> bool:
        """验证动作是否有效"""
        if self.action_type == "continuous":
            if not isinstance(action, (int, float)):
                return False
            if self.low is not None and action < self.low:
                return False
            if self.high is not None and action > self.high:
                return False
        elif self.action_type == "discrete":
            if not isinstance(action, int):
                return False
            if self.n is not None and (action < 0 or action >= self.n):
                return False
        elif self.action_type == "categorical":
            if self.categories and action not in self.categories:
                return False
        return True
    
    def normalize_action(self, action: Any) -> float:
        """归一化动作值到[0,1]或[-1,1]范围"""
        if self.action_type == "continuous":
            if self.low is not None and self.high is not None:
                # 归一化到[-1, 1]范围
                center = (self.high + self.low) / 2
                range_val = (self.high - self.low) / 2
                return (action - center) / range_val
            return float(action)
        elif self.action_type == "discrete":
            if self.n is not None:
                return (action * 2.0 / (self.n - 1)) - 1.0 if self.n > 1 else 0.0
            return float(action)
        elif self.action_type == "categorical":
            if self.categories:
                return (self.categories.index(action) * 2.0 / (len(self.categories) - 1)) - 1.0
            return 0.0
        return float(action)
    
    def clip_action(self, action: Any) -> Any:
        """裁剪动作到有效范围"""
        if self.action_type == "continuous":
            if self.low is not None:
                action = max(action, self.low)
            if self.high is not None:
                action = min(action, self.high)
            return action
        elif self.action_type == "discrete":
            if self.n is not None:
                return max(0, min(action, self.n - 1))
            return action
        return action


class ActionSpace(ABC):
    """动作空间抽象基类"""
    
    def __init__(self, space_type: ActionSpaceType, dimensions: List[ActionDimension]):
        self.space_type = space_type
        self.dimensions = {d.name: d for d in dimensions}
        self.dimension_names = [d.name for d in dimensions]
        self.n_dims = len(dimensions)
    
    @abstractmethod
    def sample(self) -> Any:
        """从动作空间中随机采样一个动作"""
        pass
    
    @abstractmethod
    def contains(self, action: Any) -> bool:
        """检查动作是否在动作空间内"""
        pass
    
    def validate_action(self, action: Any) -> bool:
        """验证动作的所有维度是否有效"""
        if self.space_type == ActionSpaceType.DISCRETE:
            return isinstance(action, int) and action >= 0
        elif self.space_type == ActionSpaceType.CONTINUOUS:
            return isinstance(action, (list, tuple, np.ndarray)) and len(action) == self.n_dims
        elif self.space_type == ActionSpaceType.MULTI_DISCRETE:
            return isinstance(action, (list, tuple, np.ndarray)) and len(action) == self.n_dims
        elif self.space_type == ActionSpaceType.HYBRID:
            return isinstance(action, dict)
        return False
    
    def get_action_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """获取各动作维度的边界"""
        bounds = {}
        for name, dimension in self.dimensions.items():
            bounds[name] = (dimension.low, dimension.high)
        return bounds
    
    def get_action_info(self) -> Dict[str, Dict[str, Any]]:
        """获取动作维度信息"""
        info = {}
        for name, dimension in self.dimensions.items():
            info[name] = {
                "type": dimension.action_type,
                "low": dimension.low,
                "high": dimension.high,
                "n": dimension.n,
                "categories": dimension.categories,
                "description": dimension.description
            }
        return info


class DiscreteActionSpace(ActionSpace):
    """离散动作空间"""
    
    def __init__(self, n: int, action_names: Optional[List[str]] = None):
        dimension = ActionDimension(
            name="action",
            action_type="discrete",
            n=n,
            description=f"离散动作空间，包含{n}个动作"
        )
        super().__init__(ActionSpaceType.DISCRETE, [dimension])
        self.n = n
        self.action_names = action_names or [f"action_{i}" for i in range(n)]
    
    def sample(self) -> int:
        """随机采样离散动作"""
        return np.random.randint(0, self.n)
    
    def contains(self, action: Any) -> bool:
        """检查动作是否在离散动作空间内"""
        return isinstance(action, int) and 0 <= action < self.n
    
    def get_action_name(self, action: int) -> str:
        """获取动作名称"""
        if 0 <= action < len(self.action_names):
            return self.action_names[action]
        return f"unknown_action_{action}"
    
    def get_action_id(self, action_name: str) -> Optional[int]:
        """根据动作名称获取动作ID"""
        try:
            return self.action_names.index(action_name)
        except ValueError:
            return None


class ContinuousActionSpace(ActionSpace):
    """连续动作空间"""
    
    def __init__(self, low: Union[float, np.ndarray], high: Union[float, np.ndarray], 
                 shape: Optional[Tuple[int, ...]] = None):
        if isinstance(low, (int, float)):
            low = np.full(shape or (1,), low)
        if isinstance(high, (int, float)):
            high = np.full(shape or (1,), high)
        
        if low.shape != high.shape:
            raise ValueError("low和high的形状必须相同")
        
        dimensions = []
        for i in range(len(low.flat)):
            dimension = ActionDimension(
                name=f"action_{i}",
                action_type="continuous",
                low=float(low.flat[i]),
                high=float(high.flat[i]),
                description=f"连续动作维度{i}，范围[{low.flat[i]}, {high.flat[i]}]"
            )
            dimensions.append(dimension)
        
        super().__init__(ActionSpaceType.CONTINUOUS, dimensions)
        self.low = low
        self.high = high
        self.shape = low.shape
    
    def sample(self) -> np.ndarray:
        """随机采样连续动作"""
        return np.random.uniform(self.low, self.high, self.shape)
    
    def contains(self, action: Any) -> bool:
        """检查动作是否在连续动作空间内"""
        if not isinstance(action, (list, tuple, np.ndarray)):
            return False
        
        action = np.array(action)
        if action.shape != self.shape:
            return False
        
        return np.all(action >= self.low) and np.all(action <= self.high)
    
    def clip(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作到有效范围"""
        return np.clip(action, self.low, self.high)


class MultiDiscreteActionSpace(ActionSpace):
    """多重离散动作空间（每个维度都是离散的）"""
    
    def __init__(self, nvec: Union[List[int], np.ndarray]):
        self.nvec = np.array(nvec)
        
        dimensions = []
        for i, n in enumerate(self.nvec):
            dimension = ActionDimension(
                name=f"action_{i}",
                action_type="discrete",
                n=int(n),
                description=f"离散动作维度{i}，包含{n}个动作"
            )
            dimensions.append(dimension)
        
        super().__init__(ActionSpaceType.MULTI_DISCRETE, dimensions)
    
    def sample(self) -> np.ndarray:
        """随机采样多重离散动作"""
        return np.array([np.random.randint(0, n) for n in self.nvec])
    
    def contains(self, action: Any) -> bool:
        """检查动作是否在多重离散动作空间内"""
        if not isinstance(action, (list, tuple, np.ndarray)):
            return False
        
        action = np.array(action)
        if action.shape != self.nvec.shape:
            return False
        
        return np.all(action >= 0) and np.all(action < self.nvec)


class HybridActionSpace(ActionSpace):
    """混合动作空间（同时包含连续和离散动作）"""
    
    def __init__(self, discrete_dims: List[ActionDimension], continuous_dims: List[ActionDimension]):
        all_dims = discrete_dims + continuous_dims
        super().__init__(ActionSpaceType.HYBRID, all_dims)
        
        self.discrete_dims = {d.name: d for d in discrete_dims}
        self.continuous_dims = {d.name: d for d in continuous_dims}
    
    def sample(self) -> Dict[str, Any]:
        """随机采样混合动作"""
        action = {}
        
        # 采样离散动作
        for name, dim in self.discrete_dims.items():
            if dim.action_type == "discrete":
                action[name] = np.random.randint(0, dim.n or 2)
            elif dim.action_type == "categorical":
                if dim.categories:
                    action[name] = np.random.choice(dim.categories)
                else:
                    action[name] = "unknown"
        
        # 采样连续动作
        for name, dim in self.continuous_dims.items():
            low = dim.low if dim.low is not None else 0.0
            high = dim.high if dim.high is not None else 1.0
            action[name] = np.random.uniform(low, high)
        
        return action
    
    def contains(self, action: Any) -> bool:
        """检查动作是否在混合动作空间内"""
        if not isinstance(action, dict):
            return False
        
        # 验证所有维度
        for name, dim in self.dimensions.items():
            if name not in action:
                return False
            if not dim.validate_action(action[name]):
                return False
        
        return True
    
    def split_action(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """将混合动作分离为离散和连续部分"""
        discrete_action = {}
        continuous_action = {}
        
        for name, value in action.items():
            if name in self.discrete_dims:
                discrete_action[name] = value
            elif name in self.continuous_dims:
                continuous_action[name] = value
        
        return discrete_action, continuous_action


class ActionSpaceFactory:
    """动作空间工厂类"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> ActionSpace:
        """从配置创建动作空间"""
        space_type = config["space_type"].lower()
        
        if space_type == "discrete":
            return DiscreteActionSpace(
                n=config["n"],
                action_names=config.get("action_names")
            )
        
        elif space_type == "continuous":
            low = config.get("low", -1.0)
            high = config.get("high", 1.0)
            shape = tuple(config["shape"]) if "shape" in config else None
            return ContinuousActionSpace(low, high, shape)
        
        elif space_type == "multi_discrete":
            return MultiDiscreteActionSpace(config["nvec"])
        
        elif space_type == "hybrid":
            discrete_dims = []
            continuous_dims = []
            
            for dim_config in config["dimensions"]:
                dim = ActionDimension(
                    name=dim_config["name"],
                    action_type=dim_config["type"],
                    low=dim_config.get("low"),
                    high=dim_config.get("high"),
                    n=dim_config.get("n"),
                    categories=dim_config.get("categories"),
                    description=dim_config.get("description")
                )
                
                if dim.action_type in ["discrete", "categorical"]:
                    discrete_dims.append(dim)
                else:
                    continuous_dims.append(dim)
            
            return HybridActionSpace(discrete_dims, continuous_dims)
        
        else:
            raise ValueError(f"不支持的动作空间类型: {space_type}")
    
    @staticmethod
    def create_simple_discrete(n: int, action_names: Optional[List[str]] = None) -> DiscreteActionSpace:
        """创建简单的离散动作空间"""
        return DiscreteActionSpace(n, action_names)
    
    @staticmethod
    def create_simple_continuous(low: float, high: float, shape: Tuple[int, ...]) -> ContinuousActionSpace:
        """创建简单的连续动作空间"""
        return ContinuousActionSpace(low, high, shape)
    
    @staticmethod
    def create_box_action_space(low: Union[float, List[float], np.ndarray],
                              high: Union[float, List[float], np.ndarray]) -> ContinuousActionSpace:
        """创建Box类型的连续动作空间"""
        if isinstance(low, (int, float)):
            low = [low]
        if isinstance(high, (int, float)):
            high = [high]
        
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        
        return ContinuousActionSpace(low, high, low.shape)