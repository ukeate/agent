"""
动作空间测试

测试不同类型的动作空间实现。
"""

import pytest
import numpy as np
from typing import Dict, Any

from apps.api.src.ai.reinforcement_learning.environment.action_space import (
    ActionSpace,
    ActionSpaceType,
    ActionDimension,
    DiscreteActionSpace,
    ContinuousActionSpace,
    MultiDiscreteActionSpace,
    HybridActionSpace,
    ActionSpaceFactory
)


class TestActionDimension:
    """动作维度测试"""
    
    def test_discrete_action_dimension(self):
        """测试离散动作维度"""
        dim = ActionDimension(
            name="move",
            action_type="discrete",
            n=4
        )
        
        # 测试有效动作
        assert dim.validate_action(0) is True
        assert dim.validate_action(3) is True
        
        # 测试无效动作
        assert dim.validate_action(-1) is False
        assert dim.validate_action(4) is False
        assert dim.validate_action("invalid") is False
        
        # 测试归一化
        assert dim.normalize_action(0) == -1.0
        assert dim.normalize_action(3) == 1.0
    
    def test_continuous_action_dimension(self):
        """测试连续动作维度"""
        dim = ActionDimension(
            name="force",
            action_type="continuous",
            low=-2.0,
            high=2.0
        )
        
        # 测试有效动作
        assert dim.validate_action(0.0) is True
        assert dim.validate_action(-2.0) is True
        assert dim.validate_action(2.0) is True
        
        # 测试无效动作
        assert dim.validate_action(-2.1) is False
        assert dim.validate_action(2.1) is False
        assert dim.validate_action("invalid") is False
        
        # 测试归一化（归一化到[-1, 1]）
        assert abs(dim.normalize_action(0.0) - 0.0) < 1e-6
        assert abs(dim.normalize_action(-2.0) - (-1.0)) < 1e-6
        assert abs(dim.normalize_action(2.0) - 1.0) < 1e-6
    
    def test_categorical_action_dimension(self):
        """测试类别动作维度"""
        dim = ActionDimension(
            name="direction",
            action_type="categorical",
            categories=["north", "south", "east", "west"]
        )
        
        # 测试有效动作
        assert dim.validate_action("north") is True
        assert dim.validate_action("west") is True
        
        # 测试无效动作
        assert dim.validate_action("northeast") is False
        assert dim.validate_action(0) is False
        
        # 测试归一化
        assert dim.normalize_action("north") == -1.0
        assert dim.normalize_action("west") == 1.0


class TestDiscreteActionSpace:
    """离散动作空间测试"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        action_space = DiscreteActionSpace(n=4)
        
        # 测试基本属性
        assert action_space.n == 4
        assert action_space.space_type == ActionSpaceType.DISCRETE
        
        # 测试采样
        for _ in range(10):
            action = action_space.sample()
            assert isinstance(action, int)
            assert 0 <= action < 4
        
        # 测试包含关系
        assert action_space.contains(0) is True
        assert action_space.contains(3) is True
        assert action_space.contains(-1) is False
        assert action_space.contains(4) is False
        assert action_space.contains("invalid") is False
    
    def test_with_action_names(self):
        """测试带动作名称的离散动作空间"""
        action_names = ["up", "down", "left", "right"]
        action_space = DiscreteActionSpace(n=4, action_names=action_names)
        
        # 测试动作名称
        assert action_space.get_action_name(0) == "up"
        assert action_space.get_action_name(3) == "right"
        assert action_space.get_action_name(5) == "unknown_action_5"
        
        # 测试动作ID获取
        assert action_space.get_action_id("up") == 0
        assert action_space.get_action_id("right") == 3
        assert action_space.get_action_id("invalid") is None


class TestContinuousActionSpace:
    """连续动作空间测试"""
    
    def test_1d_action_space(self):
        """测试一维连续动作空间"""
        action_space = ContinuousActionSpace(low=-1.0, high=1.0, shape=(1,))
        
        # 测试基本属性
        assert action_space.space_type == ActionSpaceType.CONTINUOUS
        assert action_space.shape == (1,)
        
        # 测试采样
        for _ in range(10):
            action = action_space.sample()
            assert isinstance(action, np.ndarray)
            assert action.shape == (1,)
            assert -1.0 <= action[0] <= 1.0
        
        # 测试包含关系
        assert action_space.contains([0.5]) is True
        assert action_space.contains([-1.0]) is True
        assert action_space.contains([1.0]) is True
        assert action_space.contains([-1.1]) is False
        assert action_space.contains([1.1]) is False
        assert action_space.contains([0.5, 0.5]) is False  # 错误形状
    
    def test_multi_dim_action_space(self):
        """测试多维连续动作空间"""
        low = np.array([-1.0, -2.0])
        high = np.array([1.0, 2.0])
        action_space = ContinuousActionSpace(low=low, high=high)
        
        # 测试基本属性
        assert action_space.shape == (2,)
        np.testing.assert_array_equal(action_space.low, low)
        np.testing.assert_array_equal(action_space.high, high)
        
        # 测试采样
        for _ in range(10):
            action = action_space.sample()
            assert action.shape == (2,)
            assert -1.0 <= action[0] <= 1.0
            assert -2.0 <= action[1] <= 2.0
        
        # 测试裁剪
        clipped = action_space.clip(np.array([-2.0, 3.0]))
        np.testing.assert_array_equal(clipped, [-1.0, 2.0])


class TestMultiDiscreteActionSpace:
    """多重离散动作空间测试"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        nvec = [3, 4, 2]
        action_space = MultiDiscreteActionSpace(nvec)
        
        # 测试基本属性
        assert action_space.space_type == ActionSpaceType.MULTI_DISCRETE
        np.testing.assert_array_equal(action_space.nvec, nvec)
        
        # 测试采样
        for _ in range(10):
            action = action_space.sample()
            assert isinstance(action, np.ndarray)
            assert len(action) == 3
            assert 0 <= action[0] < 3
            assert 0 <= action[1] < 4
            assert 0 <= action[2] < 2
        
        # 测试包含关系
        assert action_space.contains([0, 0, 0]) is True
        assert action_space.contains([2, 3, 1]) is True
        assert action_space.contains([3, 0, 0]) is False
        assert action_space.contains([0, 4, 0]) is False
        assert action_space.contains([0, 0, 2]) is False


class TestHybridActionSpace:
    """混合动作空间测试"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        discrete_dims = [
            ActionDimension("move", "discrete", n=4),
            ActionDimension("tool", "categorical", categories=["hammer", "wrench"])
        ]
        continuous_dims = [
            ActionDimension("force", "continuous", low=-1.0, high=1.0),
            ActionDimension("angle", "continuous", low=0.0, high=360.0)
        ]
        
        action_space = HybridActionSpace(discrete_dims, continuous_dims)
        
        # 测试基本属性
        assert action_space.space_type == ActionSpaceType.HYBRID
        assert len(action_space.discrete_dims) == 2
        assert len(action_space.continuous_dims) == 2
        
        # 测试采样
        for _ in range(10):
            action = action_space.sample()
            assert isinstance(action, dict)
            assert "move" in action
            assert "tool" in action
            assert "force" in action
            assert "angle" in action
            
            # 验证离散部分
            assert 0 <= action["move"] < 4
            assert action["tool"] in ["hammer", "wrench"]
            
            # 验证连续部分
            assert -1.0 <= action["force"] <= 1.0
            assert 0.0 <= action["angle"] <= 360.0
        
        # 测试包含关系
        valid_action = {
            "move": 1,
            "tool": "hammer",
            "force": 0.5,
            "angle": 90.0
        }
        assert action_space.contains(valid_action) is True
        
        invalid_action = {
            "move": 5,  # 超出范围
            "tool": "hammer",
            "force": 0.5,
            "angle": 90.0
        }
        assert action_space.contains(invalid_action) is False
    
    def test_split_action(self):
        """测试动作分离"""
        discrete_dims = [ActionDimension("move", "discrete", n=4)]
        continuous_dims = [ActionDimension("force", "continuous", low=-1.0, high=1.0)]
        
        action_space = HybridActionSpace(discrete_dims, continuous_dims)
        
        action = {"move": 2, "force": 0.5}
        discrete_part, continuous_part = action_space.split_action(action)
        
        assert discrete_part == {"move": 2}
        assert continuous_part == {"force": 0.5}


class TestActionSpaceFactory:
    """动作空间工厂测试"""
    
    def test_create_discrete_action_space(self):
        """测试创建离散动作空间"""
        config = {
            "space_type": "discrete",
            "n": 5,
            "action_names": ["a", "b", "c", "d", "e"]
        }
        
        action_space = ActionSpaceFactory.create_from_config(config)
        
        assert isinstance(action_space, DiscreteActionSpace)
        assert action_space.n == 5
        assert action_space.get_action_name(0) == "a"
    
    def test_create_continuous_action_space(self):
        """测试创建连续动作空间"""
        config = {
            "space_type": "continuous",
            "low": -2.0,
            "high": 2.0,
            "shape": [2]
        }
        
        action_space = ActionSpaceFactory.create_from_config(config)
        
        assert isinstance(action_space, ContinuousActionSpace)
        assert action_space.shape == (2,)
        np.testing.assert_array_equal(action_space.low, [-2.0, -2.0])
        np.testing.assert_array_equal(action_space.high, [2.0, 2.0])
    
    def test_create_multi_discrete_action_space(self):
        """测试创建多重离散动作空间"""
        config = {
            "space_type": "multi_discrete",
            "nvec": [3, 4, 2]
        }
        
        action_space = ActionSpaceFactory.create_from_config(config)
        
        assert isinstance(action_space, MultiDiscreteActionSpace)
        np.testing.assert_array_equal(action_space.nvec, [3, 4, 2])
    
    def test_create_hybrid_action_space(self):
        """测试创建混合动作空间"""
        config = {
            "space_type": "hybrid",
            "dimensions": [
                {
                    "name": "discrete_action",
                    "type": "discrete",
                    "n": 4
                },
                {
                    "name": "continuous_action",
                    "type": "continuous",
                    "low": -1.0,
                    "high": 1.0
                }
            ]
        }
        
        action_space = ActionSpaceFactory.create_from_config(config)
        
        assert isinstance(action_space, HybridActionSpace)
        assert len(action_space.discrete_dims) == 1
        assert len(action_space.continuous_dims) == 1
    
    def test_factory_methods(self):
        """测试工厂便捷方法"""
        # 测试简单离散动作空间
        discrete_space = ActionSpaceFactory.create_simple_discrete(4, ["up", "down", "left", "right"])
        assert isinstance(discrete_space, DiscreteActionSpace)
        assert discrete_space.n == 4
        
        # 测试简单连续动作空间
        continuous_space = ActionSpaceFactory.create_simple_continuous(-1.0, 1.0, (2,))
        assert isinstance(continuous_space, ContinuousActionSpace)
        assert continuous_space.shape == (2,)
        
        # 测试Box动作空间
        box_space = ActionSpaceFactory.create_box_action_space([-1.0, -2.0], [1.0, 2.0])
        assert isinstance(box_space, ContinuousActionSpace)
        assert box_space.shape == (2,)
        np.testing.assert_array_equal(box_space.low, [-1.0, -2.0])
        np.testing.assert_array_equal(box_space.high, [1.0, 2.0])
    
    def test_invalid_action_space_type(self):
        """测试无效动作空间类型"""
        config = {"space_type": "invalid"}
        
        with pytest.raises(ValueError, match="不支持的动作空间类型"):
            ActionSpaceFactory.create_from_config(config)


if __name__ == "__main__":
    pytest.main([__file__])