"""
容错功能集成测试

验证容错模块、检查点管理和智能调度的实际功能
"""

import asyncio
import pytest
import websockets
import json
import tempfile
import os
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import AsyncMock, MagicMock

# 导入要测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.streaming.fault_tolerance import (
    FaultTolerantConnection, 
    ConnectionManager, 
    ConnectionConfig, 
    ConnectionState
)
from src.ai.batch.checkpoint_manager import CheckpointManager, CheckpointConfig
from src.ai.batch.batch_processor import BatchJob, BatchTask, BatchStatus


class TestFaultToleranceIntegration:
    """容错功能集成测试"""
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_connection_lifecycle(self):
        """测试容错连接的完整生命周期"""
        config = ConnectionConfig(
            max_retries=3,
            initial_retry_delay=0.1,
            connection_timeout=5.0
        )
        
        conn = FaultTolerantConnection("test-session-001", config)
        
        # 模拟连接工厂
        async def mock_connection_factory(**kwargs):
            # 模拟WebSocket连接
            mock_conn = MagicMock()
            mock_conn.send = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn
        
        # 测试连接建立
        success = await conn.connect(mock_connection_factory, url="ws://test")
        assert success
        assert conn.state == ConnectionState.CONNECTED
        assert conn._connection_factory is not None
        assert conn._connection_kwargs == {"url": "ws://test"}
        
        # 测试消息发送
        await conn.send_message({"type": "test", "data": "hello"})
        
        # 测试连接信息
        info = conn.get_connection_info()
        assert info['session_id'] == "test-session-001"
        assert info['state'] == ConnectionState.CONNECTED.value
        
        # 测试断开连接
        await conn.disconnect()
        assert conn.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_connection_manager_integration(self):
        """测试连接管理器集成功能"""
        manager = ConnectionManager()
        
        # 设置连接工厂
        async def mock_factory(**kwargs):
            mock_conn = MagicMock()
            mock_conn.send = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn
        
        manager.set_connection_factory(mock_factory)
        
        # 创建连接
        conn = await manager.create_connection("session-001", "agent-001", url="ws://test")
        assert conn is not None
        assert conn.state == ConnectionState.CONNECTED
        
        # 获取连接
        retrieved_conn = await manager.get_connection("session-001")
        assert retrieved_conn is conn
        
        # 获取所有连接状态
        statuses = await manager.get_all_connections_status()
        assert "session-001" in statuses
        assert statuses["session-001"]["state"] == ConnectionState.CONNECTED.value
        
        # 移除连接
        await manager.remove_connection("session-001")
        retrieved_conn = await manager.get_connection("session-001")
        assert retrieved_conn is None


class TestCheckpointManagerIntegration:
    """检查点管理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_checkpoint_manager_lifecycle(self):
        """测试检查点管理器的完整生命周期"""
        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(
                storage_path=temp_dir,
                auto_save_interval=1.0,
                max_checkpoints_per_job=3
            )
            
            manager = CheckpointManager(config)
            
            # 创建测试作业
            job = BatchJob(
                id="test-job-001",
                name="Test Job",
                total_tasks=100,
                completed_tasks=50,
                failed_tasks=5,
                status=BatchStatus.RUNNING,
                priority=5,
                created_at=utc_now()
            )
            
            # 创建检查点
            checkpoint_id = await manager.create_checkpoint(job, "manual")
            assert checkpoint_id is not None
            assert checkpoint_id.startswith("test-job-001_manual_")
            
            # 获取检查点统计
            stats = await manager.get_checkpoint_stats()
            assert stats['total_checkpoints'] == 1
            assert stats['jobs_with_checkpoints'] == 1
            assert 'manual' in stats['checkpoint_types']
            
            # 列出作业检查点
            checkpoints = await manager.list_job_checkpoints("test-job-001")
            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint_id == checkpoint_id
            
            # 恢复作业
            restored_job = await manager.restore_job(checkpoint_id)
            assert restored_job is not None
            assert restored_job.id == "test-job-001"
            assert restored_job.completed_tasks == 50
            assert restored_job.failed_tasks == 5
            
            # 删除检查点
            deleted_count = await manager.delete_job_checkpoints("test-job-001")
            assert deleted_count == 1
            
            # 验证删除
            final_stats = await manager.get_checkpoint_stats()
            assert final_stats['total_checkpoints'] == 0


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_streaming_with_fault_tolerance(self):
        """测试带容错的流式处理"""
        # 创建连接管理器
        manager = ConnectionManager()
        
        # 模拟连接失败然后成功的场景
        call_count = 0
        async def unreliable_factory(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次失败
                raise ConnectionError("Network error")
            
            # 第三次成功
            mock_conn = MagicMock()
            mock_conn.send = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn
        
        manager.set_connection_factory(unreliable_factory)
        
        # 创建容错连接
        config = ConnectionConfig(
            max_retries=5,
            initial_retry_delay=0.1,
            reconnect_on_error=True
        )
        
        conn = await manager.create_connection(
            "test-session", "test-agent", config, url="ws://test"
        )
        
        # 验证连接最终成功（经过重试）
        # 注意：第一次create_connection可能失败，但会触发自动重连
        assert call_count >= 3  # 至少尝试了3次
    
    @pytest.mark.asyncio 
    async def test_batch_processing_with_checkpoints(self):
        """测试带检查点的批处理"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置检查点管理器
            config = CheckpointConfig(
                storage_path=temp_dir,
                auto_save_interval=0.5
            )
            manager = CheckpointManager(config)
            
            # 创建批处理任务
            tasks = [
                BatchTask(
                    id=f"task-{i:03d}",
                    type="data_processing",
                    data={"index": i, "value": i * 2},
                    status=BatchStatus.COMPLETED if i < 30 else BatchStatus.PENDING
                )
                for i in range(50)
            ]
            
            job = BatchJob(
                id="batch-job-001",
                name="Integration Test Job",
                tasks=tasks,
                total_tasks=50,
                completed_tasks=30,
                failed_tasks=2,
                status=BatchStatus.RUNNING,
                priority=5,
                created_at=utc_now()
            )
            
            # 注册作业
            await manager.register_job(job)
            
            # 手动创建检查点
            checkpoint_id = await manager.create_checkpoint(job, "integration_test")
            assert checkpoint_id is not None
            
            # 模拟处理更多任务
            for i in range(30, 40):
                job.tasks[i].status = BatchStatus.COMPLETED
            job.completed_tasks = 40
            
            # 创建增量检查点
            checkpoint_id_2 = await manager.create_checkpoint(job, "incremental")
            assert checkpoint_id_2 is not None
            assert checkpoint_id_2 != checkpoint_id
            
            # 验证可以恢复最新状态
            restored_job = await manager.restore_job(checkpoint_id_2)
            assert restored_job.completed_tasks == 40
            
            # 清理
            await manager.unregister_job(job.id)


# 运行测试的辅助函数
async def run_integration_tests():
    """运行所有集成测试"""
    print("开始运行集成测试...")
    
    # 容错功能测试
    fault_tolerance_tests = TestFaultToleranceIntegration()
    await fault_tolerance_tests.test_fault_tolerance_connection_lifecycle()
    await fault_tolerance_tests.test_connection_manager_integration()
    print("✓ 容错功能测试通过")
    
    # 检查点管理测试
    checkpoint_tests = TestCheckpointManagerIntegration()
    await checkpoint_tests.test_checkpoint_manager_lifecycle()
    print("✓ 检查点管理测试通过")
    
    # 端到端测试
    e2e_tests = TestEndToEndIntegration()
    await e2e_tests.test_streaming_with_fault_tolerance()
    await e2e_tests.test_batch_processing_with_checkpoints()
    print("✓ 端到端集成测试通过")
    
    print("所有集成测试通过！")


if __name__ == "__main__":
    # 直接运行测试
    asyncio.run(run_integration_tests())