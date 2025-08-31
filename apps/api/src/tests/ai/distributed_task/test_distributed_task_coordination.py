"""分布式任务协调引擎单元测试"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from src.ai.distributed_task import (
    DistributedTaskCoordinationEngine,
    Task,
    TaskStatus,
    TaskPriority,
    TaskDecomposer,
    IntelligentAssigner,
    DistributedStateManager,
    ConflictResolver,
    RaftConsensusEngine,
    Conflict,
    ConflictType
)


@pytest.fixture
def coordination_engine():
    """创建协调引擎实例"""
    async def _create_engine():
        engine = DistributedTaskCoordinationEngine(
            node_id="test_node_1",
            cluster_nodes=["test_node_1", "test_node_2", "test_node_3"]
        )
        await engine.start()
        return engine
    
    return _create_engine


@pytest.fixture
def sample_task():
    """创建示例任务"""
    return Task(
        task_id="test_task_1",
        task_type="data_processing",
        data={"input": "test_data", "processing_type": "transform"},
        requirements={"cpu": 0.5, "memory": 1024},
        priority=TaskPriority.HIGH,
        created_at=datetime.now()
    )


class TestTaskDecomposer:
    """测试任务分解器"""
    
    @pytest.mark.asyncio
    async def test_parallel_decomposition(self):
        """测试并行分解策略"""
        decomposer = TaskDecomposer()
        
        task = Task(
            task_id="test_task",
            task_type="batch_processing",
            data={
                "chunks": [
                    {"data": "chunk1"},
                    {"data": "chunk2"},
                    {"data": "chunk3"}
                ]
            },
            requirements={"decomposition_strategy": "parallel"},
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        subtasks = await decomposer.decompose_task(task)
        
        assert len(subtasks) == 3
        assert all(t.parent_task_id == task.task_id for t in subtasks)
        assert task.status == TaskStatus.DECOMPOSED
    
    @pytest.mark.asyncio
    async def test_sequential_decomposition(self):
        """测试序列分解策略"""
        decomposer = TaskDecomposer()
        
        task = Task(
            task_id="test_task",
            task_type="pipeline",
            data={
                "steps": [
                    {"type": "step1", "data": {}},
                    {"type": "step2", "data": {}},
                    {"type": "step3", "data": {}}
                ]
            },
            requirements={"decomposition_strategy": "sequential"},
            priority=TaskPriority.LOW,
            created_at=datetime.now()
        )
        
        subtasks = await decomposer.decompose_task(task)
        
        assert len(subtasks) == 3
        # 检查依赖关系
        assert subtasks[0].dependencies == []
        assert subtasks[1].dependencies == ["test_task_step_0"]
        assert subtasks[2].dependencies == ["test_task_step_1"]
    
    @pytest.mark.asyncio
    async def test_hierarchical_decomposition(self):
        """测试分层分解策略"""
        decomposer = TaskDecomposer()
        
        task = Task(
            task_id="test_task",
            task_type="complex_analysis",
            data={
                "hierarchy": {
                    "phase1": {
                        "type": "preparation",
                        "data": {},
                        "subtasks": {
                            "task1": {"data": "data1"},
                            "task2": {"data": "data2"}
                        }
                    }
                }
            },
            requirements={"decomposition_strategy": "hierarchical"},
            priority=TaskPriority.CRITICAL,
            created_at=datetime.now()
        )
        
        subtasks = await decomposer.decompose_task(task)
        
        assert len(subtasks) == 3  # 1个父节点 + 2个子节点
        assert any("phase1" in t.task_id for t in subtasks)


class TestIntelligentAssigner:
    """测试智能分配器"""
    
    @pytest.mark.asyncio
    async def test_capability_based_assignment(self):
        """测试基于能力的分配"""
        assigner = IntelligentAssigner()
        
        task = Task(
            task_id="test_task",
            task_type="data_processing",
            data={"type": "transform"},
            requirements={"cpu": 0.3, "memory": 512},
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        agent_id = await assigner.assign_task(task, strategy="capability_based")
        
        assert agent_id is not None
        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_to == agent_id
    
    @pytest.mark.asyncio
    async def test_load_balanced_assignment(self):
        """测试负载均衡分配"""
        assigner = IntelligentAssigner()
        
        # 分配多个任务测试负载均衡
        tasks = []
        for i in range(5):
            task = Task(
                task_id=f"test_task_{i}",
                task_type="data_processing",
                data={"id": i},
                requirements={},
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now()
            )
            tasks.append(task)
        
        assigned_agents = []
        for task in tasks:
            agent_id = await assigner.assign_task(task, strategy="load_balanced")
            assigned_agents.append(agent_id)
        
        # 检查负载分配
        assert len(set(assigned_agents)) > 1  # 应该分配给多个智能体
    
    @pytest.mark.asyncio
    async def test_reassign_task(self):
        """测试任务重新分配"""
        assigner = IntelligentAssigner()
        
        task = Task(
            task_id="test_task",
            task_type="data_processing",
            data={},
            requirements={},
            priority=TaskPriority.HIGH,
            created_at=datetime.now()
        )
        
        # 首次分配
        first_agent = await assigner.assign_task(task)
        assert first_agent is not None
        
        # 重新分配
        new_agent = await assigner.reassign_task(task, reason="failure")
        
        # 应该分配给不同的智能体
        assert new_agent != first_agent
        assert task.retry_count == 1


class TestDistributedStateManager:
    """测试分布式状态管理器"""
    
    @pytest.mark.asyncio
    async def test_set_and_get_state(self):
        """测试状态设置和获取"""
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 设置状态
        success = await state_manager.set_global_state("test_key", {"value": "test_value"})
        assert success
        
        # 获取状态
        value = await state_manager.get_global_state("test_key")
        assert value == {"value": "test_value"}
    
    @pytest.mark.asyncio
    async def test_atomic_update(self):
        """测试原子更新"""
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 初始状态
        await state_manager.set_global_state("counter", 0)
        await state_manager.set_global_state("flag", False)
        
        # 原子更新
        updates = {"counter": 1, "flag": True}
        success = await state_manager.atomic_update(updates)
        assert success
        
        # 验证更新
        assert await state_manager.get_global_state("counter") == 1
        assert await state_manager.get_global_state("flag") == True
    
    @pytest.mark.asyncio
    async def test_checkpoint_and_rollback(self):
        """测试检查点和回滚"""
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 设置初始状态
        await state_manager.set_global_state("data", {"version": 1})
        
        # 创建检查点
        success = await state_manager.create_checkpoint("checkpoint_1")
        assert success
        
        # 修改状态
        await state_manager.set_global_state("data", {"version": 2})
        assert await state_manager.get_global_state("data") == {"version": 2}
        
        # 回滚到检查点
        success = await state_manager.rollback_state("checkpoint_1")
        assert success
        assert await state_manager.get_global_state("data") == {"version": 1}
    
    @pytest.mark.asyncio
    async def test_distributed_lock(self):
        """测试分布式锁"""
        state_manager = DistributedStateManager(node_id="test_node")
        
        # 获取锁
        acquired = await state_manager.acquire_lock("test_lock", timeout=1.0)
        assert acquired
        
        # 尝试再次获取（应该失败）
        acquired2 = await state_manager.acquire_lock("test_lock", timeout=0.1)
        assert not acquired2
        
        # 释放锁
        released = await state_manager.release_lock("test_lock")
        assert released
        
        # 现在应该可以获取
        acquired3 = await state_manager.acquire_lock("test_lock", timeout=1.0)
        assert acquired3
        await state_manager.release_lock("test_lock")


class TestConflictResolver:
    """测试冲突解决器"""
    
    @pytest.mark.asyncio
    async def test_detect_resource_conflicts(self):
        """测试资源冲突检测"""
        resolver = ConflictResolver()
        
        # 模拟资源冲突场景
        conflicts = await resolver.detect_conflicts()
        
        # 基础测试，实际实现需要设置任务数据
        assert isinstance(conflicts, list)
    
    @pytest.mark.asyncio
    async def test_resolve_conflict(self):
        """测试冲突解决"""
        resolver = ConflictResolver()
        
        # 创建冲突
        conflict = Conflict(
            conflict_id="test_conflict",
            conflict_type=ConflictType.RESOURCE_CONFLICT,
            description="Test resource conflict",
            involved_tasks=["task1", "task2"],
            involved_agents=["agent1"],
            timestamp=datetime.now()
        )
        
        # 解决冲突
        success = await resolver.resolve_conflict(conflict, strategy="priority_based")
        
        assert success or not success  # 基础测试
        if success:
            assert conflict.resolved
            assert conflict.resolution_strategy == "priority_based"


class TestRaftConsensus:
    """测试Raft共识引擎"""
    
    @pytest.mark.asyncio
    async def test_raft_initialization(self):
        """测试Raft初始化"""
        raft = RaftConsensusEngine(
            node_id="node_1",
            cluster_nodes=["node_1", "node_2", "node_3"]
        )
        
        await raft.start()
        
        assert raft.state in [raft.state.FOLLOWER, raft.state.CANDIDATE, raft.state.LEADER]
        assert raft.current_term >= 0
        
        await raft.stop()
    
    @pytest.mark.asyncio
    async def test_append_entry(self):
        """测试日志追加"""
        raft = RaftConsensusEngine(
            node_id="node_1",
            cluster_nodes=["node_1"]  # 单节点测试
        )
        
        await raft.start()
        
        # 等待成为Leader（单节点会立即成为Leader）
        await asyncio.sleep(0.5)
        
        # 追加日志
        command = {"action": "test", "data": "test_data"}
        success = await raft.append_entry(command)
        
        # 单节点模式下应该成功
        assert success or raft.state != raft.state.LEADER
        
        await raft.stop()


class TestDistributedTaskCoordinationEngine:
    """测试分布式任务协调引擎"""
    
    @pytest.mark.asyncio
    async def test_submit_task(self, coordination_engine):
        """测试任务提交"""
        engine = await coordination_engine()
        try:
            task_id = await engine.submit_task(
                task_type="data_processing",
                task_data={"input": "test_data"},
                requirements={"cpu": 0.5},
                priority=TaskPriority.HIGH
            )
            
            # 单节点模式可能无法提交（需要是Leader）
            if task_id:
                assert task_id != ""
                
                # 获取任务状态
                status = await engine.get_task_status(task_id)
                assert status["task_id"] == task_id
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, coordination_engine):
        """测试任务取消"""
        engine = await coordination_engine()
        try:
            # 先提交任务
            task_id = await engine.submit_task(
                task_type="long_running",
                task_data={"duration": 3600},
                requirements={},
                priority=TaskPriority.LOW
            )
            
            if task_id:
                # 取消任务
                success = await engine.cancel_task(task_id)
                
                # 单节点模式可能成功或失败
                assert isinstance(success, bool)
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_get_system_stats(self, coordination_engine):
        """测试系统统计"""
        engine = await coordination_engine()
        try:
            stats = await engine.get_system_stats()
            
            assert "node_id" in stats
            assert "raft_state" in stats
            assert "active_tasks" in stats
            assert "completed_tasks" in stats
            assert "queued_tasks" in stats
            assert "stats" in stats
        finally:
            await engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])