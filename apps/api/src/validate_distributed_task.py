import asyncio
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from ai.distributed_task import (
    DistributedTaskCoordinationEngine,
    TaskPriority,
    TaskDecomposer,
    IntelligentAssigner,
    DistributedStateManager,
    ConflictResolver,
    RaftConsensusEngine
)
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python
"""分布式任务协调系统验证脚本"""

async def validate_distributed_task_system():
    """验证分布式任务协调系统"""

    logger.info("分布式任务协调系统验证")
    logger.info("验证分隔线", line="=" * 60)
    
    # 创建协调引擎
    engine = DistributedTaskCoordinationEngine(
        node_id="validator_node_1",
        cluster_nodes=["validator_node_1", "validator_node_2", "validator_node_3"]
    )
    
    try:
        # 启动引擎
        logger.info("启动协调引擎")
        await engine.start()
        logger.info("引擎已启动", node_id=engine.node_id)
        logger.info("Raft状态", state=engine.raft_consensus.state.value)
        
        # 提交简单任务
        logger.info("提交简单任务")
        task_id_1 = await engine.submit_task(
            task_type="data_processing",
            task_data={"input": "sample_data", "operation": "transform"},
            requirements={"cpu": 0.3, "memory": 512},
            priority=TaskPriority.HIGH
        )
        
        if task_id_1:
            logger.info("任务提交成功", task_id=task_id_1)
        else:
            logger.warning("任务提交失败（可能不是Leader节点）")
        
        # 提交可分解任务
        logger.info("提交可分解任务")
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
            logger.info("批处理任务提交成功", task_id=task_id_2)
        else:
            logger.warning("批处理任务提交失败")
        
        # 等待处理
        await asyncio.sleep(1)
        
        # 获取系统状态
        logger.info("系统状态统计")
        stats = await engine.get_system_stats()
        logger.info("系统状态详情", stats=stats)
        
        # 测试任务分解器
        logger.info("测试任务分解器")
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
            created_at=utc_now()
        )
        
        subtasks = await decomposer.decompose_task(test_task)
        logger.info("任务分解成功", subtask_count=len(subtasks))
        
        # 测试智能分配器
        logger.info("测试智能分配器")
        assigner = IntelligentAssigner()
        
        # 测试不同分配策略
        strategies = ["capability_based", "load_balanced", "resource_optimized", "locality_aware"]
        for strategy in strategies:
            test_task.status = test_task.status.PENDING  # 重置状态
            agent_id = await assigner.assign_task(test_task, strategy=strategy)
            if agent_id:
                logger.info("任务分配成功", strategy=strategy, agent_id=agent_id)
            else:
                logger.warning("任务分配失败", strategy=strategy)
        
        # 测试状态管理器
        logger.info("测试分布式状态管理")
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 设置和获取状态
        await state_manager.set_global_state("test_key", {"value": "test_data", "timestamp": utc_now().isoformat()})
        state = await state_manager.get_global_state("test_key")
        logger.info("状态存储成功", state=state)
        
        # 测试分布式锁
        lock_acquired = await state_manager.acquire_lock("test_resource", timeout=1.0)
        if lock_acquired:
            logger.info("分布式锁获取成功")
            await state_manager.release_lock("test_resource")
            logger.info("分布式锁释放成功")
        
        # 测试冲突解决器
        logger.info("测试冲突检测")
        resolver = ConflictResolver(state_manager=state_manager, task_coordinator=engine)
        conflicts = await resolver.detect_conflicts()
        logger.info("检测到冲突", conflict_count=len(conflicts))
        
        # 测试Raft共识
        logger.info(
            "Raft共识引擎状态",
            current_term=engine.raft_consensus.current_term,
            log_length=len(engine.raft_consensus.log),
            commit_index=engine.raft_consensus.commit_index,
        )
        
        # 测试检查点功能
        logger.info("测试状态检查点")
        checkpoint_created = await state_manager.create_checkpoint("validation_checkpoint")
        if checkpoint_created:
            logger.info("检查点创建成功")
        
        logger.info("验证分隔线", line="=" * 60)
        logger.info("分布式任务协调系统验证完成")
        logger.info("验证分隔线", line="=" * 60)
        
    except Exception:
        logger.exception("验证过程出错")
        
    finally:
        # 清理
        await engine.stop()
        logger.info("引擎已停止")

if __name__ == "__main__":
    setup_logging()
    asyncio.run(validate_distributed_task_system())
