"""
状态空间表示框架

定义和管理强化学习环境中的状态空间，支持连续和离散状态空间。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from ..qlearning.base import AgentState

class StateSpaceType(Enum):
    """状态空间类型"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous" 
    HYBRID = "hybrid"

@dataclass
class StateFeature:
    """状态特征定义"""
    name: str
    feature_type: str  # "continuous", "discrete", "categorical"
    low: Optional[float] = None  # 最小值（连续特征）
    high: Optional[float] = None  # 最大值（连续特征）
    categories: Optional[List[str]] = None  # 类别（分类特征）
    description: Optional[str] = None
    
    def validate_value(self, value: Any) -> bool:
        """验证特征值是否有效"""
        if self.feature_type == "continuous":
            if not isinstance(value, (int, float)):
                return False
            if self.low is not None and value < self.low:
                return False
            if self.high is not None and value > self.high:
                return False
        elif self.feature_type == "discrete":
            if not isinstance(value, int):
                return False
            if self.low is not None and value < self.low:
                return False
            if self.high is not None and value > self.high:
                return False
        elif self.feature_type == "categorical":
            if self.categories and value not in self.categories:
                return False
        return True
    
    def normalize_value(self, value: Any) -> float:
        """归一化特征值到[0,1]范围"""
        if self.feature_type == "continuous":
            if self.low is not None and self.high is not None:
                return (value - self.low) / (self.high - self.low)
            return float(value)
        elif self.feature_type == "discrete":
            if self.low is not None and self.high is not None:
                return (value - self.low) / (self.high - self.low)
            return float(value)
        elif self.feature_type == "categorical":
            if self.categories:
                return float(self.categories.index(value)) / len(self.categories)
            return 0.0
        return float(value)

class StateSpace(ABC):
    """状态空间抽象基类"""
    
    def __init__(self, space_type: StateSpaceType, features: List[StateFeature]):
        self.space_type = space_type
        self.features = {f.name: f for f in features}
        self.feature_names = [f.name for f in features]
        self.dimension = len(features)
    
    @abstractmethod
    def sample(self) -> AgentState:
        """从状态空间中随机采样一个状态"""
        ...
    
    @abstractmethod
    def contains(self, state: AgentState) -> bool:
        """检查状态是否在状态空间内"""
        ...
    
    def validate_state(self, state: AgentState) -> bool:
        """验证状态的所有特征是否有效"""
        if not isinstance(state.features, dict):
            return False
        
        for feature_name, feature in self.features.items():
            if feature_name not in state.features:
                return False
            value = state.features[feature_name]
            if not feature.validate_value(value):
                return False
        return True
    
    def normalize_state(self, state: AgentState) -> np.ndarray:
        """将状态归一化为数值向量"""
        normalized = []
        for feature_name in self.feature_names:
            if feature_name in state.features:
                feature = self.features[feature_name]
                normalized_value = feature.normalize_value(state.features[feature_name])
                normalized.append(normalized_value)
            else:
                normalized.append(0.0)
        return np.array(normalized, dtype=np.float32)
    
    def get_feature_bounds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """获取各特征的边界"""
        bounds = {}
        for name, feature in self.features.items():
            bounds[name] = (feature.low, feature.high)
        return bounds
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """获取特征信息"""
        info = {}
        for name, feature in self.features.items():
            info[name] = {
                "type": feature.feature_type,
                "low": feature.low,
                "high": feature.high,
                "categories": feature.categories,
                "description": feature.description
            }
        return info

class DiscreteStateSpace(StateSpace):
    """离散状态空间"""
    
    def __init__(self, features: List[StateFeature]):
        super().__init__(StateSpaceType.DISCRETE, features)
        # 验证所有特征都是离散或分类的
        for feature in features:
            if feature.feature_type not in ["discrete", "categorical"]:
                raise ValueError(f"离散状态空间不支持特征类型: {feature.feature_type}")
    
    def sample(self) -> AgentState:
        """随机采样离散状态"""
        state_features = {}
        
        for feature_name, feature in self.features.items():
            if feature.feature_type == "discrete":
                low = int(feature.low) if feature.low is not None else 0
                high = int(feature.high) if feature.high is not None else 10
                state_features[feature_name] = np.random.randint(low, high + 1)
            elif feature.feature_type == "categorical":
                if feature.categories:
                    state_features[feature_name] = np.random.choice(feature.categories)
                else:
                    state_features[feature_name] = "unknown"
        
        return AgentState.create(features=state_features)
    
    def contains(self, state: AgentState) -> bool:
        """检查状态是否在离散状态空间内"""
        return self.validate_state(state)
    
    def get_state_count(self) -> int:
        """获取离散状态总数"""
        count = 1
        for feature in self.features.values():
            if feature.feature_type == "discrete":
                low = int(feature.low) if feature.low is not None else 0
                high = int(feature.high) if feature.high is not None else 10
                count *= (high - low + 1)
            elif feature.feature_type == "categorical":
                if feature.categories:
                    count *= len(feature.categories)
                else:
                    count *= 1
        return count
    
    def enumerate_states(self) -> List[AgentState]:
        """枚举所有可能的离散状态（小状态空间使用）"""
        if self.get_state_count() > 10000:
            raise ValueError("状态空间过大，无法枚举所有状态")
        
        # 递归生成所有状态组合
        def generate_combinations(feature_idx: int, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
            if feature_idx >= len(self.feature_names):
                return [current_state.copy()]
            
            feature_name = self.feature_names[feature_idx]
            feature = self.features[feature_name]
            states = []
            
            if feature.feature_type == "discrete":
                low = int(feature.low) if feature.low is not None else 0
                high = int(feature.high) if feature.high is not None else 10
                for value in range(low, high + 1):
                    current_state[feature_name] = value
                    states.extend(generate_combinations(feature_idx + 1, current_state))
            elif feature.feature_type == "categorical":
                if feature.categories:
                    for value in feature.categories:
                        current_state[feature_name] = value
                        states.extend(generate_combinations(feature_idx + 1, current_state))
            
            return states
        
        all_state_dicts = generate_combinations(0, {})
        return [AgentState.create(features=state_dict) for state_dict in all_state_dicts]

class ContinuousStateSpace(StateSpace):
    """连续状态空间"""
    
    def __init__(self, features: List[StateFeature]):
        super().__init__(StateSpaceType.CONTINUOUS, features)
        # 验证所有特征都是连续的
        for feature in features:
            if feature.feature_type != "continuous":
                raise ValueError(f"连续状态空间不支持特征类型: {feature.feature_type}")
    
    def sample(self) -> AgentState:
        """随机采样连续状态"""
        state_features = {}
        
        for feature_name, feature in self.features.items():
            low = feature.low if feature.low is not None else 0.0
            high = feature.high if feature.high is not None else 1.0
            state_features[feature_name] = np.random.uniform(low, high)
        
        return AgentState.create(features=state_features)
    
    def contains(self, state: AgentState) -> bool:
        """检查状态是否在连续状态空间内"""
        return self.validate_state(state)
    
    def clip_state(self, state: AgentState) -> AgentState:
        """将状态裁剪到有效范围内"""
        clipped_features = {}
        
        for feature_name, value in state.features.items():
            if feature_name in self.features:
                feature = self.features[feature_name]
                clipped_value = value
                
                if feature.low is not None:
                    clipped_value = max(clipped_value, feature.low)
                if feature.high is not None:
                    clipped_value = min(clipped_value, feature.high)
                
                clipped_features[feature_name] = clipped_value
            else:
                clipped_features[feature_name] = value
        
        return AgentState.create(features=clipped_features, context=state.context)
    
    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取边界数组（用于优化算法）"""
        lows = []
        highs = []
        
        for feature_name in self.feature_names:
            feature = self.features[feature_name]
            low = feature.low if feature.low is not None else -np.inf
            high = feature.high if feature.high is not None else np.inf
            lows.append(low)
            highs.append(high)
        
        return np.array(lows), np.array(highs)

class HybridStateSpace(StateSpace):
    """混合状态空间（同时包含连续和离散特征）"""
    
    def __init__(self, features: List[StateFeature]):
        super().__init__(StateSpaceType.HYBRID, features)
        
        self.continuous_features = [f for f in features if f.feature_type == "continuous"]
        self.discrete_features = [f for f in features if f.feature_type in ["discrete", "categorical"]]
    
    def sample(self) -> AgentState:
        """随机采样混合状态"""
        state_features = {}
        
        # 采样连续特征
        for feature in self.continuous_features:
            low = feature.low if feature.low is not None else 0.0
            high = feature.high if feature.high is not None else 1.0
            state_features[feature.name] = np.random.uniform(low, high)
        
        # 采样离散特征
        for feature in self.discrete_features:
            if feature.feature_type == "discrete":
                low = int(feature.low) if feature.low is not None else 0
                high = int(feature.high) if feature.high is not None else 10
                state_features[feature.name] = np.random.randint(low, high + 1)
            elif feature.feature_type == "categorical":
                if feature.categories:
                    state_features[feature.name] = np.random.choice(feature.categories)
                else:
                    state_features[feature.name] = "unknown"
        
        return AgentState.create(features=state_features)
    
    def contains(self, state: AgentState) -> bool:
        """检查状态是否在混合状态空间内"""
        return self.validate_state(state)
    
    def split_features(self, state: AgentState) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """将状态特征分离为连续和离散部分"""
        continuous_part = {}
        discrete_part = {}
        
        for feature_name, value in state.features.items():
            if feature_name in self.features:
                feature = self.features[feature_name]
                if feature.feature_type == "continuous":
                    continuous_part[feature_name] = value
                else:
                    discrete_part[feature_name] = value
        
        return continuous_part, discrete_part

class StateSpaceFactory:
    """状态空间工厂类"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> StateSpace:
        """从配置创建状态空间"""
        features = []
        for feature_config in config["features"]:
            feature = StateFeature(
                name=feature_config["name"],
                feature_type=feature_config["type"],
                low=feature_config.get("low"),
                high=feature_config.get("high"),
                categories=feature_config.get("categories"),
                description=feature_config.get("description")
            )
            features.append(feature)
        
        space_type = config["space_type"].lower()
        if space_type == "discrete":
            return DiscreteStateSpace(features)
        elif space_type == "continuous":
            return ContinuousStateSpace(features)
        elif space_type == "hybrid":
            return HybridStateSpace(features)
        else:
            raise ValueError(f"不支持的状态空间类型: {space_type}")
    
    @staticmethod
    def create_simple_continuous(feature_names: List[str], 
                               bounds: List[Tuple[float, float]]) -> ContinuousStateSpace:
        """创建简单的连续状态空间"""
        features = []
        for name, (low, high) in zip(feature_names, bounds):
            feature = StateFeature(
                name=name,
                feature_type="continuous",
                low=low,
                high=high,
                description=f"连续特征 {name} 范围 [{low}, {high}]"
            )
            features.append(feature)
        
        return ContinuousStateSpace(features)
    
    @staticmethod  
    def create_simple_discrete(feature_names: List[str],
                             bounds: List[Tuple[int, int]]) -> DiscreteStateSpace:
        """创建简单的离散状态空间"""
        features = []
        for name, (low, high) in zip(feature_names, bounds):
            feature = StateFeature(
                name=name,
                feature_type="discrete",
                low=float(low),
                high=float(high),
                description=f"离散特征 {name} 范围 [{low}, {high}]"
            )
            features.append(feature)
        
        return DiscreteStateSpace(features)
