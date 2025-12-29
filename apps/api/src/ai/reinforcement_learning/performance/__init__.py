"""
强化学习性能优化模块

本模块提供完整的性能优化解决方案，包括：
- GPU加速训练
- 优化的经验回放缓冲区
- 分布式训练支持
- 集成测试框架
- 性能基准测试和超参数优化

主要组件：
- GPUAccelerator: GPU加速训练
- OptimizedReplayBuffer: 高性能经验回放
- DistributedTrainingManager: 分布式训练管理
- IntegrationTestSuite: 端到端集成测试
- HyperparameterOptimizer: 自动超参数优化
- PerformanceBenchmark: 性能基准测试
"""

from .gpu_accelerator import (
    GPUAccelerator,
    GPUConfig,
    GPUStrategy,
    PerformanceMonitor,
    create_optimized_gpu_config
)
from .optimized_replay_buffer import (
    OptimizedReplayBuffer,
    BufferConfig,
    BufferStrategy,
    CompressionType,
    create_optimized_buffer_config
)
from .distributed_training import (
    DistributedTrainingManager,
    DistributedConfig,
    DistributionStrategy,
    NodeType,
    create_distributed_config
)
from .integration_test import (
    IntegrationTestSuite,
    TestConfig,
    TestScenario,
    TestEnvironment,
    run_integration_tests
)
from .benchmark_optimizer import (
    HyperparameterOptimizer,
    PerformanceBenchmark,
    HyperparameterSpace,
    OptimizationTarget,
    BenchmarkConfig,
    BenchmarkMetric,
    run_hyperparameter_optimization,
    run_performance_benchmark
)
from src.ai.reinforcement_learning.performance import (
    GPUAccelerator, OptimizedReplayBuffer, create_performance_config

)

__all__ = [
    # GPU加速
    "GPUAccelerator",
    "GPUConfig", 
    "GPUStrategy",
    "PerformanceMonitor",
    "create_optimized_gpu_config",
    
    # 优化回放缓冲区
    "OptimizedReplayBuffer",
    "BufferConfig",
    "BufferStrategy", 
    "CompressionType",
    "create_optimized_buffer_config",
    
    # 分布式训练
    "DistributedTrainingManager",
    "DistributedConfig",
    "DistributionStrategy",
    "NodeType",
    "create_distributed_config",
    
    # 集成测试
    "IntegrationTestSuite",
    "TestConfig",
    "TestScenario",
    "TestEnvironment", 
    "run_integration_tests",
    
    # 基准测试和优化
    "HyperparameterOptimizer",
    "PerformanceBenchmark",
    "HyperparameterSpace",
    "OptimizationTarget", 
    "BenchmarkConfig",
    "BenchmarkMetric",
    "run_hyperparameter_optimization",
    "run_performance_benchmark"
]

# 版本信息
__version__ = "1.0.0"
__author__ = "AI Agent Development Team"

# 性能优化配置预设
PERFORMANCE_PRESETS = {
    "high_performance": {
        "gpu_config": {
            "enable_mixed_precision": True,
            "enable_xla": True,
            "batch_size_multiplier": 4
        },
        "buffer_config": {
            "strategy": "prioritized",
            "compression": True,
            "batch_size": 128,
            "num_parallel_calls": 8
        },
        "distributed_config": {
            "strategy": "data_parallel",
            "num_workers": 4,
            "compression_enabled": True
        }
    },
    
    "memory_efficient": {
        "gpu_config": {
            "enable_mixed_precision": True,
            "memory_limit": 4096,  # 4GB
            "batch_size_multiplier": 2
        },
        "buffer_config": {
            "strategy": "uniform",
            "compression": True,
            "batch_size": 32,
            "capacity": 50000
        },
        "distributed_config": {
            "strategy": "parameter_server",
            "num_workers": 2,
            "compression_enabled": True
        }
    },
    
    "development": {
        "gpu_config": {
            "enable_mixed_precision": False,
            "enable_xla": False,
            "batch_size_multiplier": 1
        },
        "buffer_config": {
            "strategy": "uniform",
            "compression": False,
            "batch_size": 32,
            "capacity": 10000
        },
        "distributed_config": {
            "strategy": "data_parallel",
            "num_workers": 1,
            "compression_enabled": False
        }
    }
}

def create_performance_config(preset: str = "high_performance") -> dict:
    """
    创建性能配置预设
    
    Args:
        preset: 预设名称 ("high_performance", "memory_efficient", "development")
    
    Returns:
        包含GPU、缓冲区和分布式配置的字典
    """
    if preset not in PERFORMANCE_PRESETS:
        raise ValueError(f"未知的预设: {preset}. 可用预设: {list(PERFORMANCE_PRESETS.keys())}")
    
    preset_config = PERFORMANCE_PRESETS[preset]
    
    return {
        "gpu_config": GPUConfig(**preset_config["gpu_config"]),
        "buffer_config": BufferConfig(**preset_config["buffer_config"]),
        "distributed_config": DistributedConfig(**preset_config["distributed_config"])
    }

def optimize_for_hardware() -> dict:
    """
    根据当前硬件自动优化配置
    
    Returns:
        优化后的配置字典
    """
    from src.core.tensorflow_config import tensorflow_lazy
    import psutil
    
    if not tensorflow_lazy.available:
        return create_performance_config("development")
    
    # 检测硬件
    tf = tensorflow_lazy.tf
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 基础配置
    if gpu_count > 0:
        if memory_gb >= 16:
            preset = "high_performance"
        else:
            preset = "memory_efficient"
    else:
        preset = "development"
    
    config = create_performance_config(preset)
    
    # 根据硬件调整
    if gpu_count > 1:
        config["distributed_config"].strategy = DistributionStrategy.MULTI_GPU
        config["distributed_config"].num_workers = min(gpu_count, 4)
    
    if cpu_count >= 8:
        config["buffer_config"].num_parallel_calls = min(cpu_count, 16)
    
    if memory_gb < 8:
        config["gpu_config"].memory_limit = 2048  # 2GB限制
        config["buffer_config"].capacity = 25000   # 减少缓冲区大小
    
    return config

# 便捷函数
def quick_performance_test(episodes: int = 100) -> dict:
    """
    快速性能测试
    
    Args:
        episodes: 测试episode数量
    
    Returns:
        性能测试结果
    """
    from .integration_test import run_integration_tests
    
    scenarios = ["basic_training", "performance_benchmark"]
    return run_integration_tests(scenarios)

def quick_optimization(target: str = "final_performance", trials: int = 20) -> dict:
    """
    快速超参数优化
    
    Args:
        target: 优化目标
        trials: 试验次数
    
    Returns:
        优化结果
    """
    return run_hyperparameter_optimization(target=target, n_trials=trials)

# 使用示例
"""
# 基础使用

# 创建高性能配置
config = create_performance_config("high_performance")
gpu_accelerator = GPUAccelerator(config["gpu_config"])
replay_buffer = OptimizedReplayBuffer(config["buffer_config"])

# 自动硬件优化
optimized_config = optimize_for_hardware()

# 快速测试
test_results = quick_performance_test()

# 快速优化
optimization_results = quick_optimization(target="convergence_speed", trials=10)
"""
