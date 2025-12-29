"""
经验回放缓冲区实现

支持随机采样、优先级采样等经验回放机制，提高样本效率和训练稳定性。
"""

import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import heapq
from dataclasses import dataclass
from .base import Experience, AgentState

@dataclass
class SumTree:
    """用于优先级采样的Sum Tree数据结构"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 完全二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.write_index = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索叶节点索引"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """获取总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """添加新经验"""
        idx = self.write_index + self.capacity - 1
        self.data[self.write_index] = data
        self.update(idx, priority)
        
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """根据累积概率获取经验"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class ReplayBuffer:
    """基础经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """添加经验到缓冲区"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样经验批次"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """缓冲区大小"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
    
    def is_ready(self, min_size: int) -> bool:
        """检查是否有足够的经验开始训练"""
        return len(self.buffer) >= min_size

class PrioritizedReplayBuffer(ReplayBuffer):
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        super().__init__(capacity)
        self.alpha = alpha  # 优先级影响程度
        self.beta = beta    # 重要性采样偏差修正
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.tree = SumTree(capacity)
        self.epsilon = 1e-6  # 防止零优先级
    
    def push(self, experience: Experience) -> None:
        """添加经验到优先级缓冲区"""
        priority = (experience.priority + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """采样经验批次，返回经验、索引和重要性采样权重"""
        if self.tree.n_entries == 0:
            return [], np.array([]), np.array([])
        
        batch_size = min(batch_size, self.tree.n_entries)
        experiences = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, experience = self.tree.get(s)
            
            experiences.append(experience)
            indices.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries

class CircularReplayBuffer(ReplayBuffer):
    """循环经验回放缓冲区 - 高效内存使用"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0
    
    def push(self, experience: Experience) -> None:
        """添加经验"""
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样"""
        if self.size < batch_size:
            return [exp for exp in self.buffer[:self.size] if exp is not None]
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self) -> None:
        self.buffer = [None] * self.capacity
        self.position = 0
        self.size = 0

class EpisodeBuffer:
    """Episode级别的经验缓冲区"""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: List[Experience] = []
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验到当前episode"""
        self.current_episode.append(experience)
    
    def end_episode(self) -> None:
        """结束当前episode"""
        if self.current_episode:
            self.episodes.append(self.current_episode.copy())
            self.current_episode.clear()
    
    def sample_episodes(self, num_episodes: int) -> List[List[Experience]]:
        """采样完整的episodes"""
        if len(self.episodes) < num_episodes:
            return list(self.episodes)
        return random.sample(self.episodes, num_episodes)
    
    def sample_experiences(self, batch_size: int) -> List[Experience]:
        """从所有episodes中随机采样经验"""
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode)
        
        if len(all_experiences) < batch_size:
            return all_experiences
        return random.sample(all_experiences, batch_size)
    
    def get_episode_rewards(self) -> List[float]:
        """获取每个episode的总奖励"""
        rewards = []
        for episode in self.episodes:
            total_reward = sum(exp.reward for exp in episode)
            rewards.append(total_reward)
        return rewards
    
    def __len__(self) -> int:
        return len(self.episodes)

class MultiStepBuffer:
    """多步学习缓冲区"""
    
    def __init__(self, capacity: int, n_steps: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = CircularReplayBuffer(capacity)
        self.n_step_buffer: deque = deque(maxlen=n_steps)
    
    def push(self, experience: Experience) -> None:
        """添加经验，自动计算n步奖励"""
        self.n_step_buffer.append(experience)
        
        if len(self.n_step_buffer) == self.n_steps:
            # 计算n步奖励
            n_step_reward = 0
            discount = 1
            
            for i, exp in enumerate(self.n_step_buffer):
                n_step_reward += discount * exp.reward
                discount *= self.gamma
                if exp.done:
                    break
            
            # 创建n步经验
            first_exp = self.n_step_buffer[0]
            last_exp = self.n_step_buffer[-1]
            
            n_step_experience = Experience.create(
                state=first_exp.state,
                action=first_exp.action,
                reward=n_step_reward,
                next_state=last_exp.next_state,
                done=last_exp.done
            )
            
            self.buffer.push(n_step_experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样n步经验"""
        return self.buffer.sample(batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer.clear()
        self.n_step_buffer.clear()
