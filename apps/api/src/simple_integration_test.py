#!/usr/bin/env python3
"""
简化的容错功能集成测试

验证关键功能模块是否能正常导入和运行
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试容错模块导入
        from ai.streaming.fault_tolerance import (
            FaultTolerantConnection,
            ConnectionManager,
            ConnectionConfig,
            ConnectionState
        )
        print("✓ 容错模块导入成功")
        
        # 测试检查点模块导入
        from ai.batch.checkpoint_manager import CheckpointManager, CheckpointConfig
        from ai.batch.batch_processor import BatchJob, BatchTask, BatchStatus
        print("✓ 检查点和批处理模块导入成功")
        
        # 测试streaming包的集成导入
        from ai.streaming import connection_manager
        print("✓ streaming包集成导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

async def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        from ai.streaming.fault_tolerance import ConnectionConfig, FaultTolerantConnection
        from ai.batch.checkpoint_manager import CheckpointManager, CheckpointConfig
        from ai.batch.batch_processor import BatchJob, BatchStatus
        from datetime import datetime
        import tempfile
        
        # 测试容错连接配置
        config = ConnectionConfig(max_retries=3, initial_retry_delay=0.1)
        conn = FaultTolerantConnection("test-session", config)
        print("✓ 容错连接创建成功")
        
        # 测试检查点管理器
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_config = CheckpointConfig(storage_path=temp_dir)
            manager = CheckpointManager(checkpoint_config)
            print("✓ 检查点管理器创建成功")
            
            # 测试统计获取
            stats = await manager.get_checkpoint_stats()
            assert isinstance(stats, dict)
            print("✓ 检查点统计获取成功")
        
        # 测试批处理作业创建
        from ai.batch.batch_processor import BatchTask
        tasks = [
            BatchTask(id=f"task-{i}", type="test", data={"index": i})
            for i in range(3)
        ]
        job = BatchJob(
            id="test-job",
            name="测试作业",
            tasks=tasks,
            status=BatchStatus.PENDING,
            priority=5,
            created_at=datetime.utcnow()
        )
        print("✓ 批处理作业创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration():
    """测试简单的集成场景"""
    print("\n测试集成场景...")
    
    try:
        from ai.streaming.fault_tolerance import ConnectionManager
        from ai.batch.checkpoint_manager import CheckpointManager, CheckpointConfig
        from ai.batch.batch_processor import BatchJob, BatchStatus
        from datetime import datetime
        import tempfile
        from unittest.mock import AsyncMock, MagicMock
        
        # 容错连接管理器
        conn_manager = ConnectionManager()
        
        # 模拟连接工厂
        async def mock_factory(**kwargs):
            mock_conn = MagicMock()
            mock_conn.send = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn
        
        conn_manager.set_connection_factory(mock_factory)
        print("✓ 连接管理器配置成功")
        
        # 检查点管理器配置
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(
                CheckpointConfig(storage_path=temp_dir)
            )
            
            # 创建测试作业
            from ai.batch.batch_processor import BatchTask
            tasks = [
                BatchTask(
                    id=f"task-{i}",
                    type="integration_test",
                    data={"index": i},
                    status=BatchStatus.COMPLETED if i < 2 else BatchStatus.PENDING
                )
                for i in range(5)
            ]
            job = BatchJob(
                id="integration-test-job",
                name="集成测试作业",
                tasks=tasks,
                completed_tasks=2,
                failed_tasks=0,
                status=BatchStatus.RUNNING,
                priority=5,
                created_at=datetime.utcnow()
            )
            
            # 创建检查点
            checkpoint_id = await checkpoint_manager.create_checkpoint(job, "test")
            if checkpoint_id:
                print("✓ 检查点创建成功")
                
                # 恢复作业
                restored_job = await checkpoint_manager.restore_job(checkpoint_id)
                if restored_job and restored_job.id == job.id:
                    print("✓ 作业恢复成功")
                else:
                    print("✗ 作业恢复失败")
                    return False
            else:
                print("✗ 检查点创建失败")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("开始容错功能集成测试\n")
    
    success = True
    
    # 测试导入
    if not await test_imports():
        success = False
    
    # 测试基本功能
    if not await test_basic_functionality():
        success = False
    
    # 测试集成
    if not await test_integration():
        success = False
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 所有测试通过！容错功能集成正常")
    else:
        print("❌ 部分测试失败，需要检查问题")
    print(f"{'='*50}")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())