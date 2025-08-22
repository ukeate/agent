"""
分布式训练框架
支持多进程、多GPU和跨节点的分布式强化学习训练
"""

import os
import time
import json
import multiprocessing as mp
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import pickle
import socket
import struct

from ..qlearning.base import Experience, QLearningAgent, QLearningConfig


class DistributionStrategy(Enum):
    """分布式策略类型"""
    DATA_PARALLEL = "data_parallel"          # 数据并行
    MODEL_PARALLEL = "model_parallel"        # 模型并行
    PARAMETER_SERVER = "parameter_server"    # 参数服务器
    ALLREDUCE = "allreduce"                 # AllReduce
    ASYNC_UPDATES = "async_updates"          # 异步更新


class NodeType(Enum):
    """节点类型"""
    CHIEF = "chief"          # 主节点
    WORKER = "worker"        # 工作节点
    PS = "parameter_server"  # 参数服务器


@dataclass
class DistributedConfig:
    """分布式配置"""
    strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL
    num_workers: int = 4
    num_ps: int = 1
    batch_size_per_worker: int = 32
    gradient_accumulation_steps: int = 1
    sync_frequency: int = 100  # 同步频率
    compression_enabled: bool = True
    bandwidth_limit: Optional[float] = None  # MB/s
    fault_tolerance: bool = True
    checkpoint_frequency: int = 1000
    
    # 网络配置
    cluster_spec: Optional[Dict[str, List[str]]] = None
    task_type: NodeType = NodeType.CHIEF
    task_index: int = 0
    
    # 性能优化
    use_xla: bool = True
    mixed_precision: bool = True
    gradient_clipping: float = 1.0


class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.strategy = None
        self.cluster_resolver = None
        
        # 进程管理
        self.workers = []
        self.parameter_servers = []
        self.communication_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # 性能监控
        self.metrics = {
            "training_times": [],
            "communication_times": [],
            "synchronization_times": [],
            "throughput": []
        }
        
        self._setup_distribution_strategy()
        self._setup_cluster()
    
    def _setup_distribution_strategy(self):
        """设置分布式策略"""
        if self.config.cluster_spec:
            self.cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
                cluster_spec=self.config.cluster_spec,
                task_type=self.config.task_type.value,
                task_id=self.config.task_index
            )
        
        if self.config.strategy == DistributionStrategy.DATA_PARALLEL:
            if self.cluster_resolver:
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy(
                    cluster_resolver=self.cluster_resolver
                )
            else:
                self.strategy = tf.distribute.MirroredStrategy()
        
        elif self.config.strategy == DistributionStrategy.PARAMETER_SERVER:
            if self.cluster_resolver:
                self.strategy = tf.distribute.experimental.ParameterServerStrategy(
                    cluster_resolver=self.cluster_resolver
                )
            else:
                raise ValueError("参数服务器策略需要集群配置")
        
        elif self.config.strategy == DistributionStrategy.ALLREDUCE:
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                )
            )
        
        print(f"设置分布式策略: {self.config.strategy.value}")
        print(f"使用设备: {self.strategy.num_replicas_in_sync} 个副本")
    
    def _setup_cluster(self):
        """设置集群配置"""
        if not self.config.cluster_spec:
            # 本地多进程配置
            self.config.cluster_spec = {
                "worker": [f"localhost:{12345 + i}" for i in range(self.config.num_workers)]
            }
            
            if self.config.strategy == DistributionStrategy.PARAMETER_SERVER:
                self.config.cluster_spec["ps"] = [
                    f"localhost:{12345 + self.config.num_workers + i}" 
                    for i in range(self.config.num_ps)
                ]
    
    def create_distributed_agent(self, agent_class, *args, **kwargs) -> QLearningAgent:
        """创建分布式智能体"""
        with self.strategy.scope():
            agent = agent_class(*args, **kwargs)
            return agent
    
    def distributed_train_step(self, agent, experiences_batch: List[Experience]) -> Dict[str, float]:
        """分布式训练步骤"""
        
        def train_step_fn(experiences):
            # 单个设备训练步骤
            metrics = {}
            total_loss = 0.0
            
            for experience in experiences:
                loss = agent.update_q_value(experience)
                if loss is not None:
                    total_loss += loss
            
            metrics["loss"] = total_loss / len(experiences) if experiences else 0.0
            return metrics
        
        # 分布式执行
        start_time = time.time()
        
        # 将经验批次分配到各个副本
        per_replica_batch_size = len(experiences_batch) // self.strategy.num_replicas_in_sync
        distributed_experiences = []
        
        for i in range(self.strategy.num_replicas_in_sync):
            start_idx = i * per_replica_batch_size
            end_idx = start_idx + per_replica_batch_size
            distributed_experiences.append(experiences_batch[start_idx:end_idx])
        
        # 执行分布式训练
        per_replica_results = self.strategy.run(train_step_fn, args=(distributed_experiences,))
        
        # 聚合结果
        aggregated_metrics = {}
        for key in per_replica_results.keys():
            aggregated_metrics[key] = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_results[key], axis=None
            ).numpy()
        
        training_time = time.time() - start_time
        self.metrics["training_times"].append(training_time)
        
        return aggregated_metrics
    
    def start_parameter_server_training(self, agent_factory: Callable, training_data_fn: Callable):
        """启动参数服务器训练"""
        if self.config.strategy != DistributionStrategy.PARAMETER_SERVER:
            raise ValueError("仅支持参数服务器策略")
        
        # 启动参数服务器进程
        for i in range(self.config.num_ps):
            ps_process = mp.Process(
                target=self._run_parameter_server,
                args=(i, agent_factory, training_data_fn)
            )
            ps_process.start()
            self.parameter_servers.append(ps_process)
        
        # 启动工作进程
        for i in range(self.config.num_workers):
            worker_process = mp.Process(
                target=self._run_worker,
                args=(i, agent_factory, training_data_fn)
            )
            worker_process.start()
            self.workers.append(worker_process)
        
        print(f"启动分布式训练: {self.config.num_workers} 工作进程, {self.config.num_ps} 参数服务器")
    
    def _run_parameter_server(self, ps_id: int, agent_factory: Callable, training_data_fn: Callable):
        """运行参数服务器"""
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": self.config.cluster_spec,
            "task": {"type": "ps", "index": ps_id}
        })
        
        # 参数服务器逻辑
        with self.strategy.scope():
            agent = agent_factory()
            
            # 参数服务器主要负责存储和更新参数
            server = tf.distribute.Server(
                tf.train.ClusterSpec(self.config.cluster_spec),
                job_name="ps",
                task_index=ps_id
            )
            server.join()
    
    def _run_worker(self, worker_id: int, agent_factory: Callable, training_data_fn: Callable):
        """运行工作进程"""
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": self.config.cluster_spec,
            "task": {"type": "worker", "index": worker_id}
        })
        
        with self.strategy.scope():
            agent = agent_factory()
            training_data = training_data_fn()
            
            step_count = 0
            for batch in training_data:
                # 执行训练步骤
                metrics = self.distributed_train_step(agent, batch)
                
                # 发送结果到主进程
                self.result_queue.put({
                    "worker_id": worker_id,
                    "step": step_count,
                    "metrics": metrics
                })
                
                step_count += 1
                
                # 检查同步
                if step_count % self.config.sync_frequency == 0:
                    self._synchronize_parameters(agent)
    
    def start_data_parallel_training(self, 
                                   agents: List[QLearningAgent], 
                                   training_data: List[List[Experience]]) -> Dict[str, List[float]]:
        """启动数据并行训练"""
        if len(agents) != len(training_data):
            raise ValueError("智能体数量必须与训练数据分区数量匹配")
        
        aggregated_metrics = {"losses": [], "rewards": []}
        
        # 使用线程池进行并行训练
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for i, (agent, data_partition) in enumerate(zip(agents, training_data)):
                future = executor.submit(self._train_agent_partition, agent, data_partition, i)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                metrics = future.result()
                for key, values in metrics.items():
                    if key in aggregated_metrics:
                        aggregated_metrics[key].extend(values)
        
        # 同步参数（平均化）
        self._synchronize_agent_parameters(agents)
        
        return aggregated_metrics
    
    def _train_agent_partition(self, agent: QLearningAgent, data_partition: List[Experience], worker_id: int) -> Dict[str, List[float]]:
        """训练智能体分区"""
        metrics = {"losses": [], "rewards": []}
        
        for experience in data_partition:
            loss = agent.update_q_value(experience)
            if loss is not None:
                metrics["losses"].append(loss)
            metrics["rewards"].append(experience.reward)
        
        return metrics
    
    def _synchronize_agent_parameters(self, agents: List[QLearningAgent]):
        """同步智能体参数"""
        if not agents:
            return
        
        start_time = time.time()
        
        # 获取所有智能体的参数
        all_weights = []
        for agent in agents:
            if hasattr(agent, 'q_network'):
                weights = agent.q_network.get_weights()
                all_weights.append(weights)
        
        if not all_weights:
            return
        
        # 计算平均参数
        avg_weights = []
        for i in range(len(all_weights[0])):
            layer_weights = [weights[i] for weights in all_weights]
            avg_weight = np.mean(layer_weights, axis=0)
            avg_weights.append(avg_weight)
        
        # 应用平均参数到所有智能体
        for agent in agents:
            if hasattr(agent, 'q_network'):
                agent.q_network.set_weights(avg_weights)
                if hasattr(agent, 'target_network'):
                    agent.target_network.set_weights(avg_weights)
        
        sync_time = time.time() - start_time
        self.metrics["synchronization_times"].append(sync_time)
    
    def _synchronize_parameters(self, agent: QLearningAgent):
        """同步参数（单个智能体）"""
        # 在实际实现中，这里会与参数服务器通信
        pass
    
    def async_parameter_update(self, agent: QLearningAgent, gradients: List[np.ndarray]):
        """异步参数更新"""
        if self.config.strategy != DistributionStrategy.ASYNC_UPDATES:
            return
        
        # 压缩梯度
        if self.config.compression_enabled:
            gradients = self._compress_gradients(gradients)
        
        # 异步发送梯度更新
        update_data = {
            "gradients": gradients,
            "agent_id": agent.agent_id,
            "timestamp": time.time()
        }
        
        try:
            self.communication_queue.put(update_data, timeout=1.0)
        except queue.Full:
            print("通信队列已满，跳过此次更新")
    
    def _compress_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """压缩梯度"""
        compressed_gradients = []
        
        for grad in gradients:
            # 简单的量化压缩
            if self.config.compression_enabled:
                # 8位量化
                grad_min, grad_max = grad.min(), grad.max()
                scale = (grad_max - grad_min) / 255.0
                quantized = np.round((grad - grad_min) / scale).astype(np.uint8)
                
                compressed_gradients.append({
                    "quantized": quantized,
                    "min": grad_min,
                    "scale": scale
                })
            else:
                compressed_gradients.append(grad)
        
        return compressed_gradients
    
    def _decompress_gradients(self, compressed_gradients: List[Dict]) -> List[np.ndarray]:
        """解压梯度"""
        decompressed_gradients = []
        
        for compressed in compressed_gradients:
            if isinstance(compressed, dict) and "quantized" in compressed:
                # 反量化
                quantized = compressed["quantized"]
                grad_min = compressed["min"]
                scale = compressed["scale"]
                grad = quantized.astype(np.float32) * scale + grad_min
                decompressed_gradients.append(grad)
            else:
                decompressed_gradients.append(compressed)
        
        return decompressed_gradients
    
    def benchmark_communication(self) -> Dict[str, float]:
        """测试通信性能"""
        print("开始通信性能测试...")
        
        # 测试数据
        test_data_sizes = [1, 10, 100, 1000]  # KB
        results = {}
        
        for size_kb in test_data_sizes:
            test_data = np.random.random((size_kb * 256,)).astype(np.float32)  # 1KB ≈ 256 float32
            
            start_time = time.time()
            
            # 模拟网络传输
            compressed = pickle.dumps(test_data)
            decompressed = pickle.loads(compressed)
            
            transfer_time = time.time() - start_time
            throughput = (len(compressed) / 1024) / transfer_time  # KB/s
            
            results[f"{size_kb}KB"] = {
                "transfer_time": transfer_time,
                "throughput": throughput,
                "compression_ratio": len(compressed) / test_data.nbytes
            }
        
        return results
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        status = {
            "strategy": self.config.strategy.value,
            "num_workers": len(self.workers),
            "num_parameter_servers": len(self.parameter_servers),
            "active_workers": sum(1 for w in self.workers if w.is_alive()),
            "active_ps": sum(1 for ps in self.parameter_servers if ps.is_alive()),
            "communication_queue_size": self.communication_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }
        
        # 性能统计
        if self.metrics["training_times"]:
            status["avg_training_time"] = np.mean(self.metrics["training_times"])
        if self.metrics["synchronization_times"]:
            status["avg_sync_time"] = np.mean(self.metrics["synchronization_times"])
        
        return status
    
    def save_checkpoint(self, agents: List[QLearningAgent], checkpoint_dir: str, step: int):
        """保存检查点"""
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存所有智能体
        for i, agent in enumerate(agents):
            agent_path = os.path.join(checkpoint_path, f"agent_{i}.json")
            agent.save_model(agent_path)
        
        # 保存分布式配置
        config_path = os.path.join(checkpoint_path, "distributed_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, agent_factory: Callable, checkpoint_path: str) -> List[QLearningAgent]:
        """加载检查点"""
        # 加载配置
        config_path = os.path.join(checkpoint_path, "distributed_config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # 创建智能体并加载模型
        agents = []
        i = 0
        while True:
            agent_path = os.path.join(checkpoint_path, f"agent_{i}.json")
            if not os.path.exists(agent_path):
                break
            
            agent = agent_factory()
            agent.load_model(agent_path)
            agents.append(agent)
            i += 1
        
        print(f"加载检查点: {checkpoint_path}, 智能体数量: {len(agents)}")
        return agents
    
    def shutdown(self):
        """关闭分布式训练"""
        print("关闭分布式训练...")
        
        # 停止所有工作进程
        for worker in self.workers:
            worker.terminate()
            worker.join(timeout=5)
        
        # 停止所有参数服务器
        for ps in self.parameter_servers:
            ps.terminate()
            ps.join(timeout=5)
        
        self.workers.clear()
        self.parameter_servers.clear()


def create_distributed_config(
    strategy: str = "data_parallel",
    num_workers: int = 4,
    cluster_spec: Optional[Dict[str, List[str]]] = None
) -> DistributedConfig:
    """创建分布式配置"""
    
    strategy_map = {
        "data_parallel": DistributionStrategy.DATA_PARALLEL,
        "model_parallel": DistributionStrategy.MODEL_PARALLEL,
        "parameter_server": DistributionStrategy.PARAMETER_SERVER,
        "allreduce": DistributionStrategy.ALLREDUCE,
        "async": DistributionStrategy.ASYNC_UPDATES
    }
    
    return DistributedConfig(
        strategy=strategy_map.get(strategy, DistributionStrategy.DATA_PARALLEL),
        num_workers=num_workers,
        cluster_spec=cluster_spec,
        batch_size_per_worker=32,
        sync_frequency=100,
        compression_enabled=True,
        fault_tolerance=True,
        use_xla=True,
        mixed_precision=True
    )