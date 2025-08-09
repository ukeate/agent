"""
检查点系统测试
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from ai.langgraph.state import MessagesState, create_initial_state
from ai.langgraph.checkpoints import (
    Checkpoint, CheckpointManager, PostgreSQLCheckpointStorage,
    CheckpointModel
)


class TestCheckpoint:
    """Checkpoint数据结构测试"""
    
    def test_checkpoint_creation(self):
        """测试检查点创建"""
        workflow_id = "test-workflow"
        state = create_initial_state(workflow_id)
        metadata = {"type": "test_checkpoint"}
        
        checkpoint = Checkpoint(
            workflow_id=workflow_id,
            state=state,
            metadata=metadata
        )
        
        assert checkpoint.workflow_id == workflow_id
        assert checkpoint.state == state
        assert checkpoint.metadata == metadata
        assert isinstance(checkpoint.created_at, datetime)
        assert checkpoint.version == 1


class TestPostgreSQLCheckpointStorage:
    """PostgreSQL检查点存储测试"""
    
    def setUp(self):
        self.storage = PostgreSQLCheckpointStorage()
        self.workflow_id = "test-workflow-123"
        self.test_state = create_initial_state(self.workflow_id)
    
    @pytest.mark.asyncio
    async def test_save_checkpoint(self):
        """测试保存检查点"""
        checkpoint = Checkpoint(
            workflow_id=self.workflow_id,
            state=self.test_state,
            metadata={"type": "test"}
        )
        
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.first.return_value = None
            mock_session_instance.commit = AsyncMock()
            
            result = await self.storage.save_checkpoint(checkpoint)
            assert result == True
            mock_session_instance.add.assert_called_once()
            mock_session_instance.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_checkpoint(self):
        """测试加载检查点"""
        checkpoint_id = "test-checkpoint-id"
        
        # Mock数据库模型
        mock_model = Mock()
        mock_model.checkpoint_id = checkpoint_id
        mock_model.workflow_id = self.workflow_id
        mock_model.state_data = self.test_state
        mock_model.metadata = {"type": "test"}
        mock_model.created_at = datetime.now()
        mock_model.version = 1
        
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_model
            
            result = await self.storage.load_checkpoint(self.workflow_id, checkpoint_id)
            
            assert result is not None
            assert result.id == checkpoint_id
            assert result.workflow_id == self.workflow_id
            assert result.state == self.test_state
            assert result.metadata == {"type": "test"}
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self):
        """测试加载不存在的检查点"""
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.first.return_value = None
            
            result = await self.storage.load_checkpoint(self.workflow_id, "nonexistent")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """测试列出检查点"""
        # Mock多个检查点
        mock_models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.checkpoint_id = f"checkpoint-{i}"
            mock_model.workflow_id = self.workflow_id
            mock_model.state_data = self.test_state
            mock_model.metadata = {"type": f"test-{i}"}
            mock_model.created_at = datetime.now() - timedelta(hours=i)
            mock_model.version = 1
            mock_models.append(mock_model)
        
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_models
            
            result = await self.storage.list_checkpoints(self.workflow_id)
            
            assert len(result) == 3
            assert all(cp.workflow_id == self.workflow_id for cp in result)
            assert result[0].id == "checkpoint-0"
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        """测试删除检查点"""
        checkpoint_id = "test-checkpoint-id"
        
        mock_model = Mock()
        mock_model.is_deleted = False
        
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_model
            mock_session_instance.commit = AsyncMock()
            
            result = await self.storage.delete_checkpoint(self.workflow_id, checkpoint_id)
            
            assert result == True
            assert mock_model.is_deleted == True
            mock_session_instance.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self):
        """测试清理旧检查点"""
        # Mock 15个检查点，保留最新10个
        mock_models = []
        for i in range(15):
            mock_model = Mock()
            mock_model.checkpoint_id = f"checkpoint-{i}"
            mock_model.workflow_id = self.workflow_id
            mock_model.created_at = datetime.now() - timedelta(hours=i)
            mock_model.is_deleted = False
            mock_models.append(mock_model)
        
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_models
            mock_session_instance.commit = AsyncMock()
            
            result = await self.storage.cleanup_old_checkpoints(self.workflow_id, keep_count=10)
            
            assert result == 5  # 删除了5个旧检查点
            # 检查前5个（最旧的）被标记为删除
            for i in range(5):
                assert mock_models[-(i+1)].is_deleted == True
            mock_session_instance.commit.assert_called_once()


class TestCheckpointManager:
    """检查点管理器测试"""
    
    def setUp(self):
        self.mock_storage = Mock()
        self.manager = CheckpointManager(self.mock_storage)
        self.workflow_id = "test-workflow"
        self.test_state = create_initial_state(self.workflow_id)
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self):
        """测试创建检查点"""
        self.mock_storage.save_checkpoint = AsyncMock(return_value=True)
        
        checkpoint = await self.manager.create_checkpoint(
            workflow_id=self.workflow_id,
            state=self.test_state,
            metadata={"type": "test"}
        )
        
        assert checkpoint.workflow_id == self.workflow_id
        assert checkpoint.state == self.test_state
        assert checkpoint.metadata == {"type": "test"}
        self.mock_storage.save_checkpoint.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_checkpoint_failure(self):
        """测试创建检查点失败"""
        self.mock_storage.save_checkpoint = AsyncMock(return_value=False)
        
        with pytest.raises(RuntimeError):
            await self.manager.create_checkpoint(
                workflow_id=self.workflow_id,
                state=self.test_state
            )
    
    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self):
        """测试从检查点恢复"""
        checkpoint_id = "test-checkpoint"
        expected_checkpoint = Checkpoint(
            id=checkpoint_id,
            workflow_id=self.workflow_id,
            state=self.test_state
        )
        
        self.mock_storage.load_checkpoint = AsyncMock(return_value=expected_checkpoint)
        
        result = await self.manager.restore_from_checkpoint(self.workflow_id, checkpoint_id)
        
        assert result == self.test_state
        self.mock_storage.load_checkpoint.assert_called_once_with(self.workflow_id, checkpoint_id)
    
    @pytest.mark.asyncio
    async def test_restore_from_nonexistent_checkpoint(self):
        """测试从不存在的检查点恢复"""
        self.mock_storage.load_checkpoint = AsyncMock(return_value=None)
        
        result = await self.manager.restore_from_checkpoint(self.workflow_id, "nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self):
        """测试获取最新检查点"""
        checkpoints = [
            Checkpoint(id="checkpoint-1", workflow_id=self.workflow_id, state=self.test_state),
            Checkpoint(id="checkpoint-2", workflow_id=self.workflow_id, state=self.test_state),
            Checkpoint(id="checkpoint-3", workflow_id=self.workflow_id, state=self.test_state),
        ]
        
        self.mock_storage.list_checkpoints = AsyncMock(return_value=checkpoints)
        
        result = await self.manager.get_latest_checkpoint(self.workflow_id)
        
        assert result == checkpoints[0]  # 第一个是最新的
        self.mock_storage.list_checkpoints.assert_called_once_with(self.workflow_id)
    
    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_empty(self):
        """测试获取最新检查点（空列表）"""
        self.mock_storage.list_checkpoints = AsyncMock(return_value=[])
        
        result = await self.manager.get_latest_checkpoint(self.workflow_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cleanup_workflow_checkpoints(self):
        """测试清理工作流检查点"""
        self.mock_storage.cleanup_old_checkpoints = AsyncMock(return_value=5)
        
        result = await self.manager.cleanup_workflow_checkpoints(self.workflow_id, keep_count=10)
        
        assert result == 5
        self.mock_storage.cleanup_old_checkpoints.assert_called_once_with(self.workflow_id, 10)


class TestCheckpointIntegration:
    """检查点集成测试"""
    
    @pytest.mark.asyncio
    async def test_checkpoint_versioning(self):
        """测试检查点版本管理"""
        storage = PostgreSQLCheckpointStorage()
        workflow_id = "versioning-test"
        state = create_initial_state(workflow_id)
        
        # Mock数据库操作
        with patch('ai.langgraph.checkpoints.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # 第一次保存 - 新检查点
            mock_session_instance.query.return_value.filter.return_value.first.return_value = None
            mock_session_instance.commit = AsyncMock()
            
            checkpoint1 = Checkpoint(workflow_id=workflow_id, state=state, version=1)
            result1 = await storage.save_checkpoint(checkpoint1)
            assert result1 == True
            
            # 第二次保存 - 更新现有检查点
            existing_model = Mock()
            existing_model.version = 1
            mock_session_instance.query.return_value.filter.return_value.first.return_value = existing_model
            
            checkpoint2 = Checkpoint(workflow_id=workflow_id, state=state, version=1)
            result2 = await storage.save_checkpoint(checkpoint2)
            assert result2 == True
            assert existing_model.version == 2  # 版本号递增
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_access(self):
        """测试并发检查点访问"""
        manager = CheckpointManager()
        workflow_id = "concurrent-test"
        state = create_initial_state(workflow_id)
        
        # Mock存储操作
        with patch.object(manager.storage, 'save_checkpoint') as mock_save, \
             patch.object(manager.storage, 'load_checkpoint') as mock_load:
            
            mock_save.return_value = asyncio.Future()
            mock_save.return_value.set_result(True)
            
            mock_checkpoint = Checkpoint(workflow_id=workflow_id, state=state)
            mock_load.return_value = asyncio.Future()
            mock_load.return_value.set_result(mock_checkpoint)
            
            # 并发创建和读取检查点
            tasks = []
            for i in range(5):
                # 创建检查点任务
                tasks.append(manager.create_checkpoint(workflow_id, state, {"batch": i}))
                # 读取检查点任务
                tasks.append(manager.restore_from_checkpoint(workflow_id, f"checkpoint-{i}"))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查没有异常
            for result in results:
                if isinstance(result, Exception):
                    pytest.fail(f"并发操作失败: {result}")
            
            # 验证调用次数
            assert mock_save.call_count == 5
            assert mock_load.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])