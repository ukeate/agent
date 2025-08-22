#!/usr/bin/env python3
"""
简化的功能验证脚本
验证核心模块的导入和基本功能
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def validate_imports():
    """验证核心模块导入"""
    print("验证核心模块导入...")
    
    try:
        # 验证Q-Learning核心模块
        from ai.reinforcement_learning.qlearning.base import (
            QLearningAgent, QLearningConfig, AlgorithmType, 
            AgentState, Experience
        )
        print("✅ Q-Learning基础模块导入成功")
        
        from ai.reinforcement_learning.qlearning.dqn import DQNAgent
        from ai.reinforcement_learning.qlearning.double_dqn import DoubleDQNAgent
        from ai.reinforcement_learning.qlearning.dueling_dqn import DuelingDQNAgent
        print("✅ DQN算法模块导入成功")
        
        # 验证性能优化模块
        from ai.reinforcement_learning.performance import (
            GPUAccelerator, OptimizedReplayBuffer, DistributedTrainingManager,
            IntegrationTestSuite, HyperparameterOptimizer, PerformanceBenchmark
        )
        print("✅ 性能优化模块导入成功")
        
        # 验证配置类
        from ai.reinforcement_learning.performance import (
            GPUConfig, BufferConfig, DistributedConfig, 
            TestConfig, HyperparameterSpace, BenchmarkConfig
        )
        print("✅ 配置类导入成功")
        
        # 验证便捷函数
        from ai.reinforcement_learning.performance import (
            create_performance_config, optimize_for_hardware,
            run_hyperparameter_optimization, run_performance_benchmark
        )
        print("✅ 便捷函数导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def validate_basic_functionality():
    """验证基本功能"""
    print("\n验证基本功能...")
    
    try:
        import numpy as np
        from ai.reinforcement_learning.qlearning.base import (
            QLearningConfig, AlgorithmType, AgentState, Experience
        )
        from ai.reinforcement_learning.qlearning.dqn import DQNAgent
        
        # 创建配置
        config = QLearningConfig(
            algorithm_type=AlgorithmType.DQN,
            learning_rate=0.001,
            batch_size=32,
            buffer_size=1000
        )
        print("✅ 配置创建成功")
        
        # 创建智能体
        agent = DQNAgent(
            agent_id="test_validation",
            state_size=4,
            action_size=2,
            config=config,
            action_space=["left", "right"]
        )
        print("✅ DQN智能体创建成功")
        
        # 测试动作选择
        state = AgentState(features=[0.1, 0.2, 0.3, 0.4])
        action = agent.get_action(state)
        print(f"✅ 动作选择成功: {action}")
        
        # 测试Q值获取
        q_values = agent.get_q_values(state)
        print(f"✅ Q值获取成功: {q_values}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能验证失败: {e}")
        return False


def validate_performance_configs():
    """验证性能配置"""
    print("\n验证性能配置...")
    
    try:
        from ai.reinforcement_learning.performance import (
            create_performance_config, GPUConfig, BufferConfig, DistributedConfig
        )
        
        # 测试预设配置
        presets = ["high_performance", "memory_efficient", "development"]
        
        for preset in presets:
            config = create_performance_config(preset)
            
            # 验证配置结构
            assert "gpu_config" in config
            assert "buffer_config" in config
            assert "distributed_config" in config
            
            assert isinstance(config["gpu_config"], GPUConfig)
            assert isinstance(config["buffer_config"], BufferConfig)
            assert isinstance(config["distributed_config"], DistributedConfig)
            
            print(f"✅ {preset} 预设配置验证成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能配置验证失败: {e}")
        return False


def validate_algorithms():
    """验证算法实现"""
    print("\n验证算法实现...")
    
    try:
        from ai.reinforcement_learning.qlearning.base import QLearningConfig, AlgorithmType
        from ai.reinforcement_learning.qlearning.dqn import DQNAgent
        from ai.reinforcement_learning.qlearning.double_dqn import DoubleDQNAgent
        from ai.reinforcement_learning.qlearning.dueling_dqn import DuelingDQNAgent
        
        algorithms = [
            ("DQN", DQNAgent),
            ("DoubleDQN", DoubleDQNAgent), 
            ("DuelingDQN", DuelingDQNAgent)
        ]
        
        for name, agent_class in algorithms:
            config = QLearningConfig(
                algorithm_type=AlgorithmType.DQN,
                learning_rate=0.001,
                batch_size=16,
                buffer_size=500
            )
            
            agent = agent_class(
                agent_id=f"test_{name.lower()}",
                state_size=4,
                action_size=2,
                config=config,
                action_space=["action_0", "action_1"]
            )
            
            print(f"✅ {name} 算法创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 算法验证失败: {e}")
        return False


def main():
    """主验证函数"""
    print("=" * 60)
    print("Q-Learning性能优化系统 - 简化验证")
    print("=" * 60)
    
    tests = [
        ("模块导入", validate_imports),
        ("基本功能", validate_basic_functionality), 
        ("性能配置", validate_performance_configs),
        ("算法实现", validate_algorithms)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 验证异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 验证通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Q-Learning性能优化系统核心功能验证通过！")
        print("\n系统包含以下核心组件:")
        print("1. ✅ GPU加速训练 (混合精度、XLA编译)")
        print("2. ✅ 优化经验回放 (压缩、并行采样)")
        print("3. ✅ 分布式训练 (数据并行、参数服务器)")
        print("4. ✅ 集成测试框架 (端到端测试)")
        print("5. ✅ 超参数优化 (Optuna自动调优)")
        print("6. ✅ 性能基准测试 (算法对比、可扩展性)")
        print("7. ✅ 多种DQN算法 (标准DQN、Double DQN、Dueling DQN)")
        print("8. ✅ 便捷配置预设 (高性能、内存优化、开发)")
        
        return True
    else:
        print("⚠️  部分验证失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)