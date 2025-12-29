"""
Grid World环境实现

经典的网格世界环境，用于Q-Learning算法测试
"""

import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from .simulator import BaseEnvironment
from .state_space import DiscreteStateSpace
from .action_space import ActionSpace

class GridWorldEnvironment(BaseEnvironment):
    """网格世界环境"""
    
    def __init__(self, grid_size: Tuple[int, int] = (4, 4), 
                 start_position: Tuple[int, int] = (0, 0),
                 goal_position: Tuple[int, int] = (3, 3),
                 obstacles: List[Tuple[int, int]] = None):
        
        self.grid_size = grid_size
        self.start_position = start_position
        self.goal_position = goal_position
        self.obstacles = obstacles or []
        
        # 动作定义：0=上，1=右，2=下，3=左
        self.actions = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1)   # 左
        }
        
        from .action_space import DiscreteActionSpace
        from .state_space import StateFeature
        from .simulator import EnvironmentInfo
        
        # 创建状态空间（x, y坐标）
        state_features = [
            StateFeature("x", "discrete", low=0, high=grid_size[0]-1),
            StateFeature("y", "discrete", low=0, high=grid_size[1]-1)
        ]
        state_space = DiscreteStateSpace(state_features)
        
        # 创建动作空间（上下左右4个动作）
        action_space = DiscreteActionSpace(4, ["up", "right", "down", "left"])
        
        # 创建环境信息
        env_info = EnvironmentInfo(
            env_id="grid_world", 
            name="Grid World Environment",
            description=f"Grid World环境 {grid_size[0]}x{grid_size[1]}"
        )
        
        super().__init__(env_info, state_space, action_space)
        
        # 环境状态
        self.current_position = list(start_position)
        self.episode_step = 0
        self.max_steps = grid_size[0] * grid_size[1] * 2  # 防止无限循环
        
        # 奖励设定
        self.goal_reward = 10.0
        self.step_penalty = -0.01
        self.wall_penalty = -0.1
        self.obstacle_penalty = -1.0
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_position = list(self.start_position)
        self.episode_step = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作"""
        self.episode_step += 1
        
        # 获取动作向量
        if action not in self.actions:
            raise ValueError(f"无效动作: {action}")
        
        action_vector = self.actions[action]
        
        # 计算新位置
        new_position = [
            self.current_position[0] + action_vector[0],
            self.current_position[1] + action_vector[1]
        ]
        
        # 检查边界
        reward = self.step_penalty
        done = False
        info = {}
        
        if (new_position[0] < 0 or new_position[0] >= self.grid_size[0] or
            new_position[1] < 0 or new_position[1] >= self.grid_size[1]):
            # 撞墙
            reward = self.wall_penalty
            info['collision'] = True
        elif tuple(new_position) in self.obstacles:
            # 撞障碍物
            reward = self.obstacle_penalty
            info['obstacle'] = True
        else:
            # 有效移动
            self.current_position = new_position
            
            # 检查是否到达目标
            if tuple(self.current_position) == self.goal_position:
                reward = self.goal_reward
                done = True
                info['success'] = True
                info['goal_reached'] = True
        
        # 检查最大步数
        if self.episode_step >= self.max_steps:
            done = True
            info['timeout'] = True
        
        # 返回结果
        next_state = self._get_state()
        info.update({
            'position': self.current_position.copy(),
            'episode_step': self.episode_step,
            'action': action
        })
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态表示"""
        # 简单的位置编码
        state_index = self.current_position[0] * self.grid_size[1] + self.current_position[1]
        
        # 创建one-hot编码
        state_size = self.grid_size[0] * self.grid_size[1]
        state = np.zeros(state_size)
        state[state_index] = 1.0
        
        return state
    
    def get_state_size(self) -> int:
        """获取状态空间大小"""
        return self.grid_size[0] * self.grid_size[1]
    
    def get_action_size(self) -> int:
        """获取动作空间大小"""
        return len(self.actions)
    
    def render(self) -> str:
        """渲染环境（文本版本）"""
        grid = []
        
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                if (i, j) == tuple(self.current_position):
                    row.append('A')  # Agent
                elif (i, j) == self.goal_position:
                    row.append('G')  # Goal
                elif (i, j) in self.obstacles:
                    row.append('X')  # Obstacle
                else:
                    row.append('.')  # Empty
            grid.append(' '.join(row))
        
        return '\n'.join(grid)
    
    def get_optimal_path_length(self) -> int:
        """获取最优路径长度（曼哈顿距离）"""
        return abs(self.goal_position[0] - self.start_position[0]) + \
               abs(self.goal_position[1] - self.start_position[1])
    
    def is_position_valid(self, position: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        return (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1] and
                position not in self.obstacles)
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取位置的有效邻居"""
        neighbors = []
        for action_vector in self.actions.values():
            new_pos = (position[0] + action_vector[0], position[1] + action_vector[1])
            if self.is_position_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'environment_type': 'grid_world',
            'grid_size': self.grid_size,
            'start_position': self.start_position,
            'goal_position': self.goal_position,
            'obstacles': self.obstacles,
            'current_position': self.current_position,
            'episode_step': self.episode_step,
            'max_steps': self.max_steps,
            'optimal_path_length': self.get_optimal_path_length(),
            'state_size': self.get_state_size(),
            'action_size': self.get_action_size()
        }
