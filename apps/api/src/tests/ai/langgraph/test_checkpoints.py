"""
LangGraph检查点和状态恢复测试（修复版）
测试工作流检查点的创建、保存、加载、恢复和清理功能
"""
import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid
import json

from src.ai.langgraph.checkpoints import (
    Checkpoint,
    CheckpointStorage,
    PostgreSQLCheckpointStorage,
    CheckpointModel
)
from src.ai.langgraph.state import (
    MessagesState,
    create_initial_state,
    serialize_state,
    deserialize_state,
    validate_state
)


@pytest.fixture
def sample_state() -> MessagesState:
    """创建示例状态"""
    return {
        "workflow_id": "test-workflow-001",
        "messages": [
            {"role": "user", "content": "开始任务"},
            {"role": "assistant", "content": "正在处理"}
        ],
        "metadata": {
            "created_at": utc_now().isoformat(),
            "status": "processing",
            "step_count": 2
        },
        "context": {
            "user_id": "user_123",
            "session_id": "session_456",
            "task_type": "code_generation"
        }
    }


@pytest.fixture
def sample_checkpoint(sample_state) -> Checkpoint:
    """创建示例检查点"""
    return Checkpoint(
        id="checkpoint-001",
        workflow_id="test-workflow-001",
        state=sample_state,
        metadata={
            "checkpoint_type": "auto",
            "trigger": "step_completion",
            "node": "process_task"
        },
        created_at=utc_now(),
        version=1
    )


@pytest.fixture
def checkpoint_storage():
    """创建检查点存储实例"""
    return PostgreSQLCheckpointStorage()


@pytest.fixture
def mock_db_session():
    """创建模拟数据库会话"""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


class TestCheckpointCreation:
    """检查点创建测试"""
    
    def test_create_checkpoint_with_minimal_data(self):
        """测试使用最小数据创建检查点"""
        checkpoint = Checkpoint()
        
        assert checkpoint.id is not None
        assert checkpoint.workflow_id == ""
        assert checkpoint.state == {}
        assert checkpoint.metadata == {}
        assert checkpoint.version == 1
        assert isinstance(checkpoint.created_at, datetime)
    
    def test_create_checkpoint_with_full_data(self, sample_state):
        """测试使用完整数据创建检查点"""
        checkpoint = Checkpoint(
            id="custom-id",
            workflow_id="workflow-123",
            state=sample_state,
            metadata={"custom": "data"},
            version=5
        )
        
        assert checkpoint.id == "custom-id"
        assert checkpoint.workflow_id == "workflow-123"
        assert checkpoint.state == sample_state
        assert checkpoint.metadata == {"custom": "data"}
        assert checkpoint.version == 5
    
    def test_checkpoint_state_validation(self, sample_state):
        """测试检查点状态验证"""
        # 有效状态
        checkpoint = Checkpoint(state=sample_state)
        assert validate_state(checkpoint.state) is True
        
        # 无效状态
        invalid_state = {"invalid": "structure"}
        checkpoint_invalid = Checkpoint(state=invalid_state)
        assert validate_state(checkpoint_invalid.state) is False


class TestCheckpointSaveLoad:
    """检查点保存和加载测试"""
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_success(self, checkpoint_storage, sample_checkpoint, mock_db_session):
        """测试成功保存检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 配置execute返回值 - 检查点不存在
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=None)
            mock_db_session.execute.return_value = mock_result
            
            result = await checkpoint_storage.save_checkpoint(sample_checkpoint)
            
            assert result is True
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_update_existing(self, checkpoint_storage, sample_checkpoint, mock_db_session):
        """测试更新已存在的检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 模拟已存在的检查点
            existing_model = Mock(spec=CheckpointModel)
            existing_model.version = 1
            existing_model.state_data = None
            existing_model.checkpoint_metadata = None
            
            # 配置execute返回值
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=existing_model)
            mock_db_session.execute.return_value = mock_result
            
            result = await checkpoint_storage.save_checkpoint(sample_checkpoint)
            
            assert result is True
            assert existing_model.state_data == sample_checkpoint.state
            assert existing_model.checkpoint_metadata == sample_checkpoint.metadata
            assert existing_model.version == 2
            mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_checkpoint_success(self, checkpoint_storage, sample_checkpoint, mock_db_session):
        """测试成功加载检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 模拟数据库返回的检查点
            mock_model = Mock(spec=CheckpointModel)
            mock_model.checkpoint_id = sample_checkpoint.id
            mock_model.workflow_id = sample_checkpoint.workflow_id
            mock_model.state_data = sample_checkpoint.state
            mock_model.checkpoint_metadata = sample_checkpoint.metadata
            mock_model.created_at = sample_checkpoint.created_at
            mock_model.version = sample_checkpoint.version
            
            # 配置execute返回值
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
            mock_db_session.execute.return_value = mock_result
            
            loaded = await checkpoint_storage.load_checkpoint(
                sample_checkpoint.workflow_id,
                sample_checkpoint.id
            )
            
            assert loaded is not None
            assert loaded.id == sample_checkpoint.id
            assert loaded.workflow_id == sample_checkpoint.workflow_id
            assert loaded.state == sample_checkpoint.state
            assert loaded.metadata == sample_checkpoint.metadata
    
    @pytest.mark.asyncio
    async def test_load_checkpoint_not_found(self, checkpoint_storage, mock_db_session):
        """测试加载不存在的检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 配置execute返回值 - 检查点不存在
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=None)
            mock_db_session.execute.return_value = mock_result
            
            loaded = await checkpoint_storage.load_checkpoint("workflow-999", "checkpoint-999")
            
            assert loaded is None


class TestCheckpointManagement:
    """检查点管理测试"""
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpoint_storage, mock_db_session):
        """测试列出所有检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 模拟多个检查点
            mock_models = []
            for i in range(3):
                model = Mock(spec=CheckpointModel)
                model.checkpoint_id = f"checkpoint-{i}"
                model.workflow_id = "workflow-001"
                model.state_data = {"index": i}
                model.checkpoint_metadata = {}
                model.created_at = utc_now() - timedelta(hours=i)
                model.version = 1
                mock_models.append(model)
            
            # 配置execute返回值
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all = MagicMock(return_value=mock_models)
            mock_result.scalars = MagicMock(return_value=mock_scalars)
            mock_db_session.execute.return_value = mock_result
            
            checkpoints = await checkpoint_storage.list_checkpoints("workflow-001")
            
            assert len(checkpoints) == 3
            # 验证按时间降序排列
            for i, checkpoint in enumerate(checkpoints):
                assert checkpoint.id == f"checkpoint-{i}"
                assert checkpoint.state == {"index": i}
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpoint_storage, mock_db_session):
        """测试删除检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 模拟找到要删除的检查点
            mock_model = Mock(spec=CheckpointModel)
            mock_model.is_deleted = False
            
            # 配置execute返回值
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
            mock_db_session.execute.return_value = mock_result
            
            result = await checkpoint_storage.delete_checkpoint("workflow-001", "checkpoint-001")
            
            assert result is True
            assert mock_model.is_deleted is True
            mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, checkpoint_storage, mock_db_session):
        """测试清理旧检查点"""
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # 模拟15个检查点
            mock_models = []
            for i in range(15):
                model = Mock(spec=CheckpointModel)
                model.id = f"db-id-{i}"
                model.checkpoint_id = f"checkpoint-{i}"
                model.is_deleted = False
                model.created_at = utc_now() - timedelta(hours=i)
                mock_models.append(model)
            
            # 配置execute返回值
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all = MagicMock(return_value=mock_models)
            mock_result.scalars = MagicMock(return_value=mock_scalars)
            mock_db_session.execute.return_value = mock_result
            
            # 清理，保留10个
            deleted_count = await checkpoint_storage.cleanup_old_checkpoints("workflow-001", keep_count=10)
            
            assert deleted_count == 5
            # 验证最旧的5个被标记为删除
            for i in range(10, 15):
                assert mock_models[i].is_deleted is True
            # 验证最新的10个保留
            for i in range(10):
                assert mock_models[i].is_deleted is False


class TestCheckpointErrorHandling:
    """检查点错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_database_error(self, checkpoint_storage, sample_checkpoint):
        """测试保存检查点时的数据库错误"""
        from sqlalchemy.exc import SQLAlchemyError
        
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            # 模拟数据库连接错误
            mock_get_db.return_value.__aenter__.side_effect = SQLAlchemyError("Database connection failed")
            
            result = await checkpoint_storage.save_checkpoint(sample_checkpoint)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_load_checkpoint_database_error(self, checkpoint_storage):
        """测试加载检查点时的数据库错误"""
        from sqlalchemy.exc import SQLAlchemyError
        
        with patch('src.ai.langgraph.checkpoints.get_db_session') as mock_get_db:
            # 模拟数据库读取错误
            mock_get_db.return_value.__aenter__.side_effect = SQLAlchemyError("Database read error")
            
            result = await checkpoint_storage.load_checkpoint("workflow-001", "checkpoint-001")
            
            assert result is None