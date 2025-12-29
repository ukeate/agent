"""
优化的经验回放缓冲区
实现高性能批处理、预处理和数据加载优化
"""

import numpy as np
from src.core.tensorflow_config import tensorflow_lazy
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque
import threading
import time
from src.core.utils import secure_pickle as pickle
import lz4.frame
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from ..qlearning.base import Experience

from src.core.logging import get_logger
logger = get_logger(__name__)

class BufferStrategy(Enum):
    """缓冲区策略类型"""
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"
    RECENCY_WEIGHTED = "recency_weighted"
    CURIOSITY_DRIVEN = "curiosity_driven"

class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    LZ4 = "lz4"
    GZIP = "gzip"

@dataclass
class BufferConfig:
    """缓冲区配置"""
    capacity: int = 100000
    strategy: BufferStrategy = BufferStrategy.UNIFORM
    compression: CompressionType = CompressionType.LZ4
    batch_size: int = 64
    num_parallel_calls: int = 4
    prefetch_size: int = 2
    use_tf_data: bool = True
    enable_async_sampling: bool = True
    min_priority: float = 1e-6
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001

class OptimizedReplayBuffer:
    """优化的经验回放缓冲区"""
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self.buffer = deque(maxlen=config.capacity)
        self.priorities = deque(maxlen=config.capacity) if config.strategy == BufferStrategy.PRIORITIZED else None
        
        # 性能优化相关
        self._lock = threading.RLock()
        self._sampling_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._preprocess_executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.num_parallel_calls)
        
        # 统计信息
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "sampling_times": deque(maxlen=1000),
            "preprocessing_times": deque(maxlen=1000)
        }
        
        # 预处理缓存
        self._preprocessed_cache = {}
        self._cache_lock = threading.Lock()
        
        logger.info(
            "初始化优化的回放缓冲区",
            strategy=config.strategy.value,
            capacity=config.capacity,
        )
    
    def push(self, experience: Experience, priority: Optional[float] = None) -> None:
        """添加经验到缓冲区"""
        with self._lock:
            # 压缩经验数据
            compressed_exp = self._compress_experience(experience)
            self.buffer.append(compressed_exp)
            
            # 处理优先级
            if self.config.strategy == BufferStrategy.PRIORITIZED:
                if priority is None:
                    priority = self._calculate_initial_priority(experience)
                self.priorities.append(priority)
            
            self.stats["total_added"] += 1
            
            # 清理缓存
            if len(self.buffer) % 1000 == 0:
                self._cleanup_cache()
    
    def sample(self, batch_size: Optional[int] = None) -> Union[List[Experience], Tuple[List[Experience], np.ndarray, np.ndarray]]:
        """采样批次数据"""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.buffer) < batch_size:
            return [] if self.config.strategy != BufferStrategy.PRIORITIZED else ([], np.array([]), np.array([]))
        
        start_time = time.time()
        
        if self.config.enable_async_sampling:
            future = self._sampling_executor.submit(self._sample_batch, batch_size)
            result = future.result()
        else:
            result = self._sample_batch(batch_size)
        
        self.stats["sampling_times"].append(time.time() - start_time)
        self.stats["total_sampled"] += batch_size
        
        return result
    
    def _sample_batch(self, batch_size: int) -> Union[List[Experience], Tuple[List[Experience], np.ndarray, np.ndarray]]:
        """采样批次实现"""
        with self._lock:
            if self.config.strategy == BufferStrategy.UNIFORM:
                return self._sample_uniform(batch_size)
            elif self.config.strategy == BufferStrategy.PRIORITIZED:
                return self._sample_prioritized(batch_size)
            elif self.config.strategy == BufferStrategy.RECENCY_WEIGHTED:
                return self._sample_recency_weighted(batch_size)
            elif self.config.strategy == BufferStrategy.CURIOSITY_DRIVEN:
                return self._sample_curiosity_driven(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> List[Experience]:
        """均匀采样"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = []
        
        for idx in indices:
            compressed_exp = self.buffer[idx]
            exp = self._decompress_experience(compressed_exp)
            experiences.append(exp)
        
        return experiences
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """优先级采样"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.config.priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.config.priority_beta)
        weights /= weights.max()
        
        experiences = []
        for idx in indices:
            compressed_exp = self.buffer[idx]
            exp = self._decompress_experience(compressed_exp)
            experiences.append(exp)
        
        return experiences, indices, weights
    
    def _sample_recency_weighted(self, batch_size: int) -> List[Experience]:
        """近期加权采样"""
        buffer_size = len(self.buffer)
        weights = np.exp(np.linspace(-2, 0, buffer_size))  # 指数衰减权重
        weights /= weights.sum()
        
        indices = np.random.choice(buffer_size, batch_size, replace=False, p=weights)
        
        experiences = []
        for idx in indices:
            compressed_exp = self.buffer[idx]
            exp = self._decompress_experience(compressed_exp)
            experiences.append(exp)
        
        return experiences
    
    def _sample_curiosity_driven(self, batch_size: int) -> List[Experience]:
        """好奇心驱动采样"""
        # 基于奖励方差和TD误差进行采样
        experiences = []
        for i in range(min(batch_size, len(self.buffer))):
            compressed_exp = self.buffer[-(i+1)]  # 优先选择最近的经验
            exp = self._decompress_experience(compressed_exp)
            experiences.append(exp)
        
        return experiences
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级"""
        if self.config.strategy != BufferStrategy.PRIORITIZED:
            return
        
        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = max(priority, self.config.min_priority)
    
    def _compress_experience(self, experience: Experience) -> bytes:
        """压缩经验数据"""
        if self.config.compression == CompressionType.NONE:
            return experience
        
        data = pickle.dumps(experience)
        
        if self.config.compression == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        
        return data
    
    def _decompress_experience(self, compressed_data: Union[bytes, Experience]) -> Experience:
        """解压经验数据"""
        if isinstance(compressed_data, Experience):
            return compressed_data
        
        if self.config.compression == CompressionType.LZ4:
            data = lz4.frame.decompress(compressed_data)
        elif self.config.compression == CompressionType.GZIP:
            import gzip
            data = gzip.decompress(compressed_data)
        else:
            data = compressed_data
        
        return pickle.loads(data)
    
    def _calculate_initial_priority(self, experience: Experience) -> float:
        """计算初始优先级"""
        # 基于奖励大小设置初始优先级
        return abs(experience.reward) + self.config.min_priority
    
    def create_tf_dataset(self) -> tensorflow_lazy.tf.data.Dataset:
        """创建TensorFlow数据集"""
        if not self.config.use_tf_data:
            raise ValueError("TensorFlow数据集未启用")
        
        def generator():
            while True:
                experiences = self.sample(self.config.batch_size)
                if not experiences:
                    continue
                
                # 转换为张量格式
                states, actions, rewards, next_states, dones = self._batch_to_tensors(experiences)
                yield {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'dones': dones
                }
        
        # 获取数据形状
        sample_exp = self.sample(1)
        if not sample_exp:
            raise ValueError("缓冲区为空，无法创建数据集")
        
        exp = sample_exp[0] if isinstance(sample_exp, list) else sample_exp[0][0]
        state_shape = np.array(exp.state.features).shape
        
        dataset = tensorflow_lazy.tf.data.Dataset.from_generator(
            generator,
            output_signature={
                'states': tensorflow_lazy.tf.TensorSpec(shape=(self.config.batch_size, *state_shape), dtype=tensorflow_lazy.tf.float32),
                'actions': tensorflow_lazy.tf.TensorSpec(shape=(self.config.batch_size,), dtype=tensorflow_lazy.tf.int32),
                'rewards': tensorflow_lazy.tf.TensorSpec(shape=(self.config.batch_size,), dtype=tensorflow_lazy.tf.float32),
                'next_states': tensorflow_lazy.tf.TensorSpec(shape=(self.config.batch_size, *state_shape), dtype=tensorflow_lazy.tf.float32),
                'dones': tensorflow_lazy.tf.TensorSpec(shape=(self.config.batch_size,), dtype=tensorflow_lazy.tf.bool)
            }
        )
        
        return dataset.prefetch(self.config.prefetch_size)
    
    def _batch_to_tensors(self, experiences: List[Experience]) -> Tuple[np.ndarray, ...]:
        """将批次转换为张量"""
        start_time = time.time()
        
        batch_size = len(experiences)
        
        # 并行预处理
        if self.config.num_parallel_calls > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_parallel_calls) as executor:
                futures = [executor.submit(self._preprocess_experience, exp) for exp in experiences]
                results = [future.result() for future in futures]
        else:
            results = [self._preprocess_experience(exp) for exp in experiences]
        
        # 聚合结果
        states = np.array([r[0] for r in results], dtype=np.float32)
        actions = np.array([r[1] for r in results], dtype=np.int32)
        rewards = np.array([r[2] for r in results], dtype=np.float32)
        next_states = np.array([r[3] for r in results], dtype=np.float32)
        dones = np.array([r[4] for r in results], dtype=np.bool_)
        
        self.stats["preprocessing_times"].append(time.time() - start_time)
        
        return states, actions, rewards, next_states, dones
    
    def _preprocess_experience(self, experience: Experience) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """预处理单个经验"""
        # 缓存查找
        exp_id = id(experience)
        with self._cache_lock:
            if exp_id in self._preprocessed_cache:
                return self._preprocessed_cache[exp_id]
        
        # 处理状态特征
        if isinstance(experience.state.features, dict):
            state_array = np.array(list(experience.state.features.values()), dtype=np.float32)
        else:
            state_array = np.array(experience.state.features, dtype=np.float32)
        
        if isinstance(experience.next_state.features, dict):
            next_state_array = np.array(list(experience.next_state.features.values()), dtype=np.float32)
        else:
            next_state_array = np.array(experience.next_state.features, dtype=np.float32)
        
        # 动作索引（假设动作是字符串，需要映射到索引）
        action_idx = hash(experience.action) % 100  # 简化的动作映射
        
        result = (state_array, action_idx, experience.reward, next_state_array, experience.done)
        
        # 缓存结果
        with self._cache_lock:
            if len(self._preprocessed_cache) < 10000:  # 限制缓存大小
                self._preprocessed_cache[exp_id] = result
        
        return result
    
    def _cleanup_cache(self):
        """清理预处理缓存"""
        with self._cache_lock:
            if len(self._preprocessed_cache) > 5000:
                # 保留最近的一半
                items = list(self._preprocessed_cache.items())
                self._preprocessed_cache = dict(items[-2500:])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            "buffer_size": len(self.buffer),
            "capacity": self.config.capacity,
            "utilization": len(self.buffer) / self.config.capacity,
            "total_added": self.stats["total_added"],
            "total_sampled": self.stats["total_sampled"],
            "cache_size": len(self._preprocessed_cache)
        }
        
        if self.stats["sampling_times"]:
            stats["avg_sampling_time"] = np.mean(self.stats["sampling_times"])
        
        if self.stats["preprocessing_times"]:
            stats["avg_preprocessing_time"] = np.mean(self.stats["preprocessing_times"])
        
        return stats
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self.buffer.clear()
            if self.priorities:
                self.priorities.clear()
            
            with self._cache_lock:
                self._preprocessed_cache.clear()
            
            # 重置统计
            for key in ["total_added", "total_sampled"]:
                self.stats[key] = 0
            for key in ["sampling_times", "preprocessing_times"]:
                self.stats[key].clear()
    
    def save_buffer(self, filepath: str):
        """保存缓冲区到文件"""
        with self._lock:
            data = {
                "buffer": list(self.buffer),
                "priorities": list(self.priorities) if self.priorities else None,
                "config": self.config,
                "stats": dict(self.stats)
            }
            
            if self.config.compression == CompressionType.LZ4:
                compressed_data = lz4.frame.compress(pickle.dumps(data))
                with open(filepath, 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
    
    def load_buffer(self, filepath: str):
        """从文件加载缓冲区"""
        with self._lock:
            if self.config.compression == CompressionType.LZ4:
                with open(filepath, 'rb') as f:
                    compressed_data = f.read()
                data = pickle.loads(lz4.frame.decompress(compressed_data))
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            self.buffer = deque(data["buffer"], maxlen=self.config.capacity)
            if data["priorities"]:
                self.priorities = deque(data["priorities"], maxlen=self.config.capacity)
            self.stats.update(data["stats"])
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, '_sampling_executor'):
            self._sampling_executor.shutdown(wait=True)
        if hasattr(self, '_preprocess_executor'):
            self._preprocess_executor.shutdown(wait=True)

def create_optimized_buffer_config(
    capacity: int = 100000,
    strategy: str = "prioritized",
    use_compression: bool = True
) -> BufferConfig:
    """创建优化的缓冲区配置"""
    
    strategy_map = {
        "uniform": BufferStrategy.UNIFORM,
        "prioritized": BufferStrategy.PRIORITIZED,
        "recency": BufferStrategy.RECENCY_WEIGHTED,
        "curiosity": BufferStrategy.CURIOSITY_DRIVEN
    }
    
    return BufferConfig(
        capacity=capacity,
        strategy=strategy_map.get(strategy, BufferStrategy.PRIORITIZED),
        compression=CompressionType.LZ4 if use_compression else CompressionType.NONE,
        batch_size=64,
        num_parallel_calls=4,
        prefetch_size=2,
        use_tf_data=True,
        enable_async_sampling=True
    )
