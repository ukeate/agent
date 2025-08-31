#!/usr/bin/env python
"""分布式任务协调系统验证脚本"""

import asyncio
import json
from datetime import datetime
from ai.distributed_task import (
    DistributedTaskCoordinationEngine,
    TaskPriority,
    TaskDecomposer,
    IntelligentAssigner,
    DistributedStateManager,
    ConflictResolver,
    RaftConsensusEngine
)


async def validate_distributed_task_system():
    """验证分布式任务协调系统"""
    
    print("=" * 60)
    print("分布式任务协调系统验证")
    print("=" * 60)
    
    # 创建协调引擎
    engine = DistributedTaskCoordinationEngine(
        node_id="validator_node_1",
        cluster_nodes=["validator_node_1", "validator_node_2", "validator_node_3"]
    )
    
    try:
        # 启动引擎
        print("\n1. 启动协调引擎...")
        await engine.start()
        print(f"   ✓ 引擎已启动，节点ID: {engine.node_id}")
        print(f"   ✓ Raft状态: {engine.raft_consensus.state.value}")
        
        # 提交简单任务
        print("\n2. 提交简单任务...")
        task_id_1 = await engine.submit_task(
            task_type="data_processing",
            task_data={"input": "sample_data", "operation": "transform"},
            requirements={"cpu": 0.3, "memory": 512},
            priority=TaskPriority.HIGH
        )
        
        if task_id_1:
            print(f"   ✓ 任务提交成功: {task_id_1}")
        else:
            print(f"   ⚠ 任务提交失败（可能不是Leader节点）")
        
        # 提交可分解任务
        print("\n3. 提交可分解任务...")
        task_id_2 = await engine.submit_task(
            task_type="batch_processing",
            task_data={
                "chunks": [
                    {"data": f"chunk_{i}", "size": 100}
                    for i in range(3)
                ]
            },
            requirements={
                "decompose": True,
                "decomposition_strategy": "parallel",
                "cpu": 0.5,
                "memory": 1024
            },
            priority=TaskPriority.MEDIUM
        )
        
        if task_id_2:
            print(f"   ✓ 批处理任务提交成功: {task_id_2}")
        else:
            print(f"   ⚠ 批处理任务提交失败")
        
        # 等待处理
        await asyncio.sleep(1)
        
        # 获取系统状态
        print("\n4. 系统状态统计...")
        stats = await engine.get_system_stats()
        print(f"   节点ID: {stats['node_id']}")
        print(f"   Raft状态: {stats['raft_state']}")
        print(f"   活跃任务: {stats['active_tasks']}")
        print(f"   完成任务: {stats['completed_tasks']}")
        print(f"   排队任务: {stats['queued_tasks']}")
        print(f"   提交总数: {stats['stats']['tasks_submitted']}")
        
        # 测试任务分解器
        print("\n5. 测试任务分解器...")
        decomposer = TaskDecomposer()
        
        # 测试并行分解
        from ai.distributed_task.models import Task
        test_task = Task(
            task_id="test_decompose",
            task_type="parallel_job",
            data={
                "chunks": [
                    {"id": 1, "data": "chunk1"},
                    {"id": 2, "data": "chunk2"},
                    {"id": 3, "data": "chunk3"}
                ]
            },
            requirements={"decomposition_strategy": "parallel"},
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        subtasks = await decomposer.decompose_task(test_task)
        print(f"   ✓ 任务分解成功，生成 {len(subtasks)} 个子任务")
        
        # 测试智能分配器
        print("\n6. 测试智能分配器...")
        assigner = IntelligentAssigner()
        
        # 测试不同分配策略
        strategies = ["capability_based", "load_balanced", "resource_optimized", "locality_aware"]
        for strategy in strategies:
            test_task.status = test_task.status.PENDING  # 重置状态
            agent_id = await assigner.assign_task(test_task, strategy=strategy)
            if agent_id:
                print(f"   ✓ {strategy}: 分配给 {agent_id}")
            else:
                print(f"   ⚠ {strategy}: 分配失败")
        
        # 测试状态管理器
        print("\n7. 测试分布式状态管理...")
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 设置和获取状态
        await state_manager.set_global_state("test_key", {"value": "test_data", "timestamp": datetime.now().isoformat()})
        state = await state_manager.get_global_state("test_key")
        print(f"   ✓ 状态存储成功: {state}")
        
        # 测试分布式锁
        lock_acquired = await state_manager.acquire_lock("test_resource", timeout=1.0)
        if lock_acquired:
            print(f"   ✓ 分布式锁获取成功")
            await state_manager.release_lock("test_resource")
            print(f"   ✓ 分布式锁释放成功")
        
        # 测试冲突解决器
        print("\n8. 测试冲突检测...")
        resolver = ConflictResolver(state_manager=state_manager, task_coordinator=engine)
        conflicts = await resolver.detect_conflicts()
        print(f"   检测到 {len(conflicts)} 个冲突")
        
        # 测试Raft共识
        print("\n9. 测试Raft共识引擎...")
        print(f"   当前任期: {engine.raft_consensus.current_term}")
        print(f"   日志长度: {len(engine.raft_consensus.log)}")
        print(f"   已提交索引: {engine.raft_consensus.commit_index}")
        
        # 测试检查点功能
        print("\n10. 测试状态检查点...")
        checkpoint_created = await state_manager.create_checkpoint("validation_checkpoint")
        if checkpoint_created:
            print(f"   ✓ 检查点创建成功")
        
        print("\n" + "=" * 60)
        print("✅ 分布式任务协调系统验证完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        await engine.stop()
        print("\n引擎已停止")


if __name__ == "__main__":
    asyncio.run(validate_distributed_task_system())