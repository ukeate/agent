import sys
import time
import numpy as np
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
Q-Learning性能优化系统验证脚本
验证GPU加速、分布式训练、集成测试等核心功能
"""

sys.path.append(str(Path(__file__).parent))

def test_gpu_accelerator():
    """测试GPU加速器"""
    logger.info("="*60)
    logger.info("测试GPU加速器")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import GPUAccelerator, GPUConfig
        
        config = GPUConfig(
            enable_mixed_precision=True,
            enable_xla=True,
            batch_size_multiplier=2
        )
        
        accelerator = GPUAccelerator(config)
        
        # 获取设备信息
        device_info = accelerator.get_device_info()
        logger.info(f"检测到 {len(device_info['gpu_devices'])} 个GPU设备")
        logger.info(f"CPU核心数: {device_info['cpu_count']}")
        
        # 创建性能监控器
        monitor = accelerator.create_performance_monitor()
        monitor.start_monitoring()
        
        time.sleep(2)  # 监控2秒
        
        monitor.stop_monitoring()
        summary = monitor.get_performance_summary()
        logger.info(f"性能监控摘要: {summary}")
        
        logger.info("✅ GPU加速器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU加速器测试失败: {e}")
        return False

def test_optimized_replay_buffer():
    """测试优化的回放缓冲区"""
    logger.info("\n" + "="*60)
    logger.info("测试优化的回放缓冲区")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import OptimizedReplayBuffer, BufferConfig
        from ai.reinforcement_learning.qlearning.base import Experience, AgentState
        
        config = BufferConfig(
            capacity=1000,
            strategy="prioritized",
            compression=True,
            batch_size=32
        )
        
        buffer = OptimizedReplayBuffer(config)
        
        # 添加经验
        logger.info("添加1000个经验到缓冲区...")
        for i in range(1000):
            state = AgentState(features=np.random.random(8).tolist())
            next_state = AgentState(features=np.random.random(8).tolist())
            
            experience = Experience(
                state=state,
                action=f"action_{i % 4}",
                reward=np.random.random(),
                next_state=next_state,
                done=np.random.random() < 0.1
            )
            
            buffer.push(experience)
        
        # 采样测试
        logger.info("测试采样性能...")
        start_time = time.time()
        
        for _ in range(100):
            if hasattr(buffer, 'sample'):
                batch = buffer.sample(32)
            else:
                break
        
        sampling_time = time.time() - start_time
        logger.info(f"100次采样耗时: {sampling_time:.3f}秒")
        
        # 获取统计信息
        stats = buffer.get_statistics()
        logger.info(f"缓冲区统计: {stats}")
        
        logger.info("✅ 优化回放缓冲区测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 优化回放缓冲区测试失败: {e}")
        return False

def test_distributed_training():
    """测试分布式训练"""
    logger.info("\n" + "="*60)
    logger.info("测试分布式训练")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import DistributedTrainingManager, DistributedConfig
        
        config = DistributedConfig(
            strategy="data_parallel",
            num_workers=2,
            batch_size_per_worker=16
        )
        
        manager = DistributedTrainingManager(config)
        
        # 获取集群状态
        status = manager.get_cluster_status()
        logger.info(f"集群状态: {status}")
        
        # 测试通信性能
        comm_results = manager.benchmark_communication()
        logger.info(f"通信性能测试: {comm_results}")
        
        logger.info("✅ 分布式训练测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 分布式训练测试失败: {e}")
        return False

def test_integration_framework():
    """测试集成测试框架"""
    logger.info("\n" + "="*60)
    logger.info("测试集成测试框架")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import (
            IntegrationTestSuite, TestConfig, TestScenario
        )
        
        # 运行基础训练测试
        config = TestConfig(
            scenarios=[TestScenario.BASIC_TRAINING],
            episodes=50,  # 减少episode数量以加快测试
            save_results=False
        )
        
        test_suite = IntegrationTestSuite(config)
        results = test_suite.run_all_tests()
        
        logger.info(f"集成测试结果: {results}")
        
        # 检查测试是否通过
        basic_training = results.get("basic_training", {})
        if basic_training.get("status") == "completed":
            logger.info("✅ 集成测试框架测试通过")
            return True
        else:
            logger.error("❌ 集成测试框架测试失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 集成测试框架测试失败: {e}")
        return False

def test_benchmark_optimizer():
    """测试基准测试和优化器"""
    logger.info("\n" + "="*60)
    logger.info("测试基准测试和优化器")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import (
            PerformanceBenchmark, BenchmarkConfig, 
            HyperparameterOptimizer, HyperparameterSpace, OptimizationTarget
        )
        
        # 快速基准测试
        config = BenchmarkConfig(
            episodes_per_run=50,
            num_seeds=2,
            save_detailed_logs=False
        )
        
        benchmark = PerformanceBenchmark(config)
        
        # 运行算法对比测试
        logger.info("运行算法对比基准测试...")
        algorithm_results = benchmark.run_algorithm_comparison()
        
        logger.info("算法对比结果:")
        for algo, result in algorithm_results.items():
            performance = result.get("mean_final_performance", 0)
            logger.info(f"  {algo}: 平均性能 {performance:.3f}")
        
        # 快速超参数优化测试
        logger.info("\n运行快速超参数优化...")
        hyperparameter_space = HyperparameterSpace()
        optimization_target = OptimizationTarget.FINAL_PERFORMANCE
        
        optimizer = HyperparameterOptimizer(
            hyperparameter_space=hyperparameter_space,
            optimization_target=optimization_target,
            benchmark_config=BenchmarkConfig(episodes_per_run=25, num_seeds=1)
        )
        
        # 只运行3次试验进行快速测试
        optimization_results = optimizer.optimize(n_trials=3)
        
        logger.info(f"优化结果: 最佳目标值 {optimization_results['best_value']:.3f}")
        logger.info(f"最佳参数: {optimization_results['best_params']}")
        
        logger.info("✅ 基准测试和优化器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 基准测试和优化器测试失败: {e}")
        return False

def test_performance_presets():
    """测试性能预设配置"""
    logger.info("\n" + "="*60)
    logger.info("测试性能预设配置")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.performance import (
            create_performance_config, optimize_for_hardware, quick_performance_test
        )
        
        # 测试预设配置
        presets = ["high_performance", "memory_efficient", "development"]
        
        for preset in presets:
            config = create_performance_config(preset)
            logger.info(f"{preset} 预设配置创建成功")
            
            # 验证配置结构
            required_keys = ["gpu_config", "buffer_config", "distributed_config"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"缺少配置键: {key}")
        
        # 测试硬件优化
        logger.info("测试硬件自适应优化...")
        optimized_config = optimize_for_hardware()
        logger.info("硬件优化配置创建成功")
        
        logger.info("✅ 性能预设配置测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 性能预设配置测试失败: {e}")
        return False

def test_q_learning_integration():
    """测试Q-Learning完整集成"""
    logger.info("\n" + "="*60)
    logger.info("测试Q-Learning完整集成")
    logger.info("="*60)
    
    try:
        from ai.reinforcement_learning.qlearning.dqn import DQNAgent
        from ai.reinforcement_learning.qlearning.base import QLearningConfig, AlgorithmType
        from ai.reinforcement_learning.performance import GPUAccelerator, OptimizedReplayBuffer
        from ai.reinforcement_learning.performance import GPUConfig, BufferConfig
        
        # 创建优化配置
        gpu_config = GPUConfig(enable_mixed_precision=False, enable_xla=False)  # 简化配置
        accelerator = GPUAccelerator(gpu_config)
        
        buffer_config = BufferConfig(capacity=1000, batch_size=32)
        replay_buffer = OptimizedReplayBuffer(buffer_config)
        
        # 创建DQN智能体
        agent_config = QLearningConfig(
            algorithm_type=AlgorithmType.DQN,
            learning_rate=0.001,
            batch_size=32,
            buffer_size=1000
        )
        
        agent = DQNAgent(
            agent_id="test_agent",
            state_size=8,
            action_size=4,
            config=agent_config,
            action_space=["action_0", "action_1", "action_2", "action_3"]
        )
        
        logger.info(f"智能体创建成功: {agent.agent_id}")
        logger.info(f"网络摘要:\n{agent.get_network_summary()}")
        
        # 测试基本训练步骤
        from ai.reinforcement_learning.qlearning.base import Experience, AgentState
        
        state = AgentState(features=np.random.random(8).tolist())
        action = agent.get_action(state)
        logger.info(f"智能体动作选择: {action}")
        
        next_state = AgentState(features=np.random.random(8).tolist())
        experience = Experience(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False
        )
        
        loss = agent.update_q_value(experience)
        logger.info(f"训练损失: {loss}")
        
        logger.info("✅ Q-Learning完整集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ Q-Learning完整集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始Q-Learning性能优化系统验证")
    logger.info("="*80)
    
    tests = [
        ("GPU加速器", test_gpu_accelerator),
        ("优化回放缓冲区", test_optimized_replay_buffer),
        ("分布式训练", test_distributed_training),
        ("集成测试框架", test_integration_framework),
        ("基准测试和优化器", test_benchmark_optimizer),
        ("性能预设配置", test_performance_presets),
        ("Q-Learning完整集成", test_q_learning_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info("\n" + "="*80)
    logger.info("测试结果汇总")
    logger.info("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 所有性能优化功能验证通过！")
        return True
    else:
        logger.error("⚠️  部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    setup_logging()
    success = main()
    sys.exit(0 if success else 1)
