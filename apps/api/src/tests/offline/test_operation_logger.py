"""
操作日志系统测试
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from uuid import uuid4

from ...offline.operation_logger import OperationLogger, OperationType, Operation
from ...offline.state_manager import StateManager, SnapshotType, CompressionType
from ...offline.models import OfflineDatabase
from ...models.schemas.offline import VectorClock, SyncOperation, SyncOperationType


class TestOperationLogger:
    """操作日志记录器测试"""
    
    @pytest.fixture
    def temp_database(self):
        """临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            database = OfflineDatabase(temp_file.name)
            yield database
            database.close()
    
    @pytest.fixture
    def operation_logger(self, temp_database):
        """操作日志记录器"""
        return OperationLogger(temp_database)
    
    def test_log_operation(self, operation_logger):
        """测试记录操作"""
        session_id = "test_session"
        vector_clock = VectorClock(node_id=session_id)
        
        # 记录操作
        operation_id = operation_logger.log_operation(
            session_id=session_id,
            operation_type=OperationType.CREATE,
            entity_type="user",
            entity_id="user_123",
            data={"name": "Alice", "email": "alice@example.com"},
            vector_clock=vector_clock,
            user_id="test_user",
            metadata={"source": "api"}
        )
        
        # 验证操作ID
        assert operation_id is not None
        assert len(operation_id) > 0
        
        # 获取操作记录
        operations = operation_logger.get_session_operations(session_id, limit=10)
        
        # 验证操作记录
        assert len(operations) == 1
        operation = operations[0]
        assert operation.id == operation_id
        assert operation.session_id == session_id
        assert operation.operation_type == OperationType.CREATE
        assert operation.entity_type == "user"
        assert operation.entity_id == "user_123"
        assert operation.data["name"] == "Alice"
        assert operation.user_id == "test_user"
        assert operation.metadata["source"] == "api"
    
    def test_multiple_operations(self, operation_logger):
        """测试多个操作记录"""
        session_id = "multi_session"
        vector_clock = VectorClock(node_id=session_id)
        
        operations_data = [
            {
                "operation_type": OperationType.CREATE,
                "entity_type": "post",
                "entity_id": "post_1",
                "data": {"title": "First Post", "content": "Hello World"}
            },
            {
                "operation_type": OperationType.UPDATE,
                "entity_type": "post",
                "entity_id": "post_1",
                "data": {"content": "Updated Content"}
            },
            {
                "operation_type": OperationType.DELETE,
                "entity_type": "post",
                "entity_id": "post_1",
                "data": {}
            }
        ]
        
        # 记录多个操作
        operation_ids = []
        for op_data in operations_data:
            vector_clock.increment()
            op_id = operation_logger.log_operation(
                session_id=session_id,
                vector_clock=vector_clock,
                **op_data
            )
            operation_ids.append(op_id)
        
        # 获取所有操作
        operations = operation_logger.get_session_operations(session_id)
        
        # 验证操作数量和顺序
        assert len(operations) == 3
        assert operations[0].operation_type == OperationType.CREATE
        assert operations[1].operation_type == OperationType.UPDATE
        assert operations[2].operation_type == OperationType.DELETE
        
        # 验证向量时钟递增
        for i in range(1, len(operations)):
            prev_clock = operations[i-1].vector_clock.clock.get(session_id, 0)
            curr_clock = operations[i].vector_clock.clock.get(session_id, 0)
            assert curr_clock > prev_clock
    
    def test_operation_to_sync_operation_conversion(self, operation_logger):
        """测试操作转换为同步操作"""
        session_id = "conversion_test"
        vector_clock = VectorClock(node_id=session_id)
        
        # 测试不同操作类型的转换
        test_cases = [
            (OperationType.CREATE, SyncOperationType.PUT),
            (OperationType.UPDATE, SyncOperationType.PATCH),
            (OperationType.DELETE, SyncOperationType.DELETE),
            (OperationType.READ, SyncOperationType.PUT),
        ]
        
        for op_type, expected_sync_type in test_cases:
            operation = Operation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=op_type,
                entity_type="test_entity",
                entity_id="test_123",
                data={"test": "data"},
                timestamp=datetime.utcnow(),
                vector_clock=vector_clock
            )
            
            sync_op = operation_logger._operation_to_sync_operation(operation)
            
            assert sync_op.operation_type == expected_sync_type
            assert sync_op.session_id == operation.session_id
            assert sync_op.table_name == operation.entity_type
            assert sync_op.object_id == operation.entity_id
            assert sync_op.data == operation.data
    
    def test_sync_operation_to_operation_conversion(self, operation_logger):
        """测试同步操作转换为操作"""
        session_id = "sync_conversion_test"
        vector_clock = VectorClock(node_id=session_id)
        
        # 创建同步操作
        sync_op = SyncOperation(
            id=str(uuid4()),
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="user",
            object_id="user_456",
            object_type="user",
            data={"name": "Bob"},
            client_timestamp=datetime.utcnow(),
            vector_clock=vector_clock,
            metadata={"user_id": "bob_user"}
        )
        
        operation = operation_logger._sync_operation_to_operation(sync_op)
        
        assert operation.session_id == sync_op.session_id
        assert operation.operation_type == OperationType.UPDATE
        assert operation.entity_type == sync_op.table_name
        assert operation.entity_id == sync_op.object_id
        assert operation.data == sync_op.data
        assert operation.user_id == sync_op.metadata["user_id"]
    
    def test_operation_retrieval_with_limit(self, operation_logger):
        """测试限制条件下的操作检索"""
        session_id = "limit_test"
        vector_clock = VectorClock(node_id=session_id)
        
        # 记录多个操作
        for i in range(20):
            vector_clock.increment()
            operation_logger.log_operation(
                session_id=session_id,
                operation_type=OperationType.CREATE,
                entity_type="item",
                entity_id=f"item_{i}",
                data={"index": i},
                vector_clock=vector_clock
            )
        
        # 测试限制检索
        operations_10 = operation_logger.get_session_operations(session_id, limit=10)
        operations_5 = operation_logger.get_session_operations(session_id, limit=5)
        
        assert len(operations_10) == 10
        assert len(operations_5) == 5
        
        # 验证返回的是最新的操作
        assert operations_5[-1].data["index"] == 19  # 最新的操作
        assert operations_5[0].data["index"] == 15   # 从第15个开始


class TestStateManager:
    """状态管理器测试"""
    
    @pytest.fixture
    def temp_database(self):
        """临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            database = OfflineDatabase(temp_file.name)
            yield database
            database.close()
    
    @pytest.fixture
    def state_manager(self, temp_database):
        """状态管理器"""
        return StateManager(temp_database)
    
    def test_create_manual_snapshot(self, state_manager):
        """测试创建手动快照"""
        session_id = "snapshot_test"
        state_data = {
            "user.123": {"name": "Alice", "age": 30},
            "user.456": {"name": "Bob", "age": 25}
        }
        
        # 创建快照
        snapshot_id = state_manager.create_snapshot(
            session_id=session_id,
            state_data=state_data,
            snapshot_type=SnapshotType.MANUAL
        )
        
        assert snapshot_id is not None
        
        # 恢复快照
        restored_state = state_manager.restore_from_snapshot(
            session_id, SnapshotType.MANUAL
        )
        
        assert restored_state is not None
        assert restored_state == state_data
        assert state_manager.get_current_state(session_id) == state_data
    
    def test_state_compression(self, state_manager):
        """测试状态压缩"""
        session_id = "compression_test"
        
        # 创建大状态数据
        large_state_data = {}
        for i in range(100):
            large_state_data[f"item.{i}"] = {
                "id": i,
                "name": f"Item {i}",
                "description": "A very long description " * 20,
                "data": list(range(50))
            }
        
        # 创建压缩快照
        snapshot_id = state_manager.create_snapshot(
            session_id=session_id,
            state_data=large_state_data,
            snapshot_type=SnapshotType.MANUAL,
            compression_type=CompressionType.JSON_GZIP
        )
        
        assert snapshot_id is not None
        
        # 恢复并验证数据完整性
        restored_state = state_manager.restore_from_snapshot(
            session_id, SnapshotType.MANUAL
        )
        
        assert restored_state is not None
        assert len(restored_state) == 100
        assert restored_state["item.0"]["name"] == "Item 0"
        assert len(restored_state["item.0"]["data"]) == 50
    
    def test_state_updates(self, state_manager):
        """测试状态更新"""
        session_id = "update_test"
        
        # 初始状态更新
        update_id_1 = state_manager.update_state(
            session_id=session_id,
            entity_type="user",
            entity_id="123",
            updates={"name": "Alice", "age": 30},
            operation_type="create"
        )
        
        # 验证状态
        current_state = state_manager.get_current_state(session_id)
        assert current_state is not None
        assert current_state["user.123"]["name"] == "Alice"
        assert current_state["user.123"]["age"] == 30
        
        # 更新状态
        update_id_2 = state_manager.update_state(
            session_id=session_id,
            entity_type="user",
            entity_id="123",
            updates={"age": 31, "email": "alice@example.com"},
            operation_type="update"
        )
        
        # 验证更新后的状态
        current_state = state_manager.get_current_state(session_id)
        assert current_state["user.123"]["name"] == "Alice"  # 保持不变
        assert current_state["user.123"]["age"] == 31        # 更新
        assert current_state["user.123"]["email"] == "alice@example.com"  # 新增
        
        # 删除状态
        update_id_3 = state_manager.update_state(
            session_id=session_id,
            entity_type="user",
            entity_id="123",
            updates={},
            operation_type="delete"
        )
        
        # 验证删除
        current_state = state_manager.get_current_state(session_id)
        assert "user.123" not in current_state
    
    def test_operation_replay(self, state_manager):
        """测试操作重放"""
        session_id = "replay_test"
        
        # 模拟添加一些同步操作到数据库
        vector_clock = VectorClock(node_id=session_id)
        
        operations = [
            SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PUT,
                table_name="user",
                object_id="123",
                object_type="user",
                data={"name": "Alice", "age": 30},
                client_timestamp=datetime.utcnow() - timedelta(minutes=10),
                vector_clock=vector_clock
            ),
            SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PATCH,
                table_name="user",
                object_id="123",
                object_type="user",
                patch_data={"age": 31},
                client_timestamp=datetime.utcnow() - timedelta(minutes=5),
                vector_clock=vector_clock
            ),
            SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PUT,
                table_name="user",
                object_id="456",
                object_type="user",
                data={"name": "Bob", "age": 25},
                client_timestamp=datetime.utcnow(),
                vector_clock=vector_clock
            )
        ]
        
        # 添加操作到数据库
        for op in operations:
            state_manager.database.add_operation(op)
        
        # 重放操作
        replay_result = state_manager.replay_operations(session_id)
        
        # 验证重放结果
        assert replay_result["applied_operations_count"] == 3
        
        replayed_state = replay_result["replayed_state"]
        assert "user.123" in replayed_state
        assert "user.456" in replayed_state
        assert replayed_state["user.123"]["name"] == "Alice"
        assert replayed_state["user.123"]["age"] == 31  # 应用了PATCH更新
        assert replayed_state["user.456"]["name"] == "Bob"
    
    def test_state_diff_calculation(self, state_manager):
        """测试状态差异计算"""
        session_id = "diff_test"
        
        # 创建初始状态
        initial_state = {
            "user.123": {"name": "Alice", "age": 30},
            "user.456": {"name": "Bob", "age": 25}
        }
        
        snapshot_id_1 = state_manager.create_snapshot(
            session_id=session_id,
            state_data=initial_state,
            snapshot_type=SnapshotType.MANUAL
        )
        
        # 修改状态
        updated_state = {
            "user.123": {"name": "Alice", "age": 31},  # 修改
            "user.789": {"name": "Charlie", "age": 35}  # 新增
            # user.456 被删除
        }
        
        snapshot_id_2 = state_manager.create_snapshot(
            session_id=session_id,
            state_data=updated_state,
            snapshot_type=SnapshotType.MANUAL
        )
        
        # 计算差异
        diff = state_manager.get_state_diff(
            session_id=session_id,
            from_snapshot=snapshot_id_1,
            to_snapshot=snapshot_id_2
        )
        
        # 验证差异
        assert len(diff["added"]) == 1
        assert "user.789" in diff["added"]
        
        assert len(diff["modified"]) == 1
        assert "user.123" in diff["modified"]
        assert diff["modified"]["user.123"]["old"]["age"] == 30
        assert diff["modified"]["user.123"]["new"]["age"] == 31
        
        assert len(diff["deleted"]) == 1
        assert "user.456" in diff["deleted"]
        
        assert diff["total_changes"] == 3
    
    def test_auto_snapshot_creation(self, state_manager):
        """测试自动快照创建"""
        session_id = "auto_snapshot_test"
        
        # 设置较短的自动快照间隔用于测试
        state_manager.auto_snapshot_interval = timedelta(seconds=1)
        
        # 设置初始状态
        state_manager.current_states[session_id] = {
            "user.123": {"name": "Alice", "age": 30}
        }
        
        # 第一次调用应该创建快照
        snapshot_id_1 = state_manager.create_auto_snapshot(session_id)
        assert snapshot_id_1 is not None
        
        # 立即再次调用应该不创建快照
        snapshot_id_2 = state_manager.create_auto_snapshot(session_id)
        assert snapshot_id_2 is None
        
        # 等待足够长时间后应该创建新快照
        import time
        time.sleep(1.1)
        
        snapshot_id_3 = state_manager.create_auto_snapshot(session_id)
        assert snapshot_id_3 is not None
        assert snapshot_id_3 != snapshot_id_1
    
    def test_state_statistics(self, state_manager):
        """测试状态统计信息"""
        session_id = "stats_test"
        
        # 创建一些状态更新
        state_manager.update_state(
            session_id, "user", "123", {"name": "Alice"}, "create"
        )
        state_manager.update_state(
            session_id, "user", "456", {"name": "Bob"}, "create"
        )
        state_manager.update_state(
            session_id, "post", "1", {"title": "Hello"}, "create"
        )
        
        # 获取统计信息
        stats = state_manager.get_state_statistics(session_id)
        
        # 验证统计信息
        assert stats["total_entities"] == 3
        assert stats["entity_types"]["user"] == 2
        assert stats["entity_types"]["post"] == 1
        assert stats["total_updates"] == 3
        assert stats["last_update"] is not None
    
    def test_state_export_import(self, state_manager):
        """测试状态导出导入"""
        session_id = "export_import_test"
        
        # 创建初始状态
        original_state = {
            "user.123": {"name": "Alice", "age": 30},
            "user.456": {"name": "Bob", "age": 25}
        }
        state_manager.current_states[session_id] = original_state
        
        # 导出状态
        exported_data = state_manager.export_state(session_id, include_metadata=True)
        
        # 验证导出数据
        assert exported_data["state"] == original_state
        assert exported_data["format"] == "json"
        assert "export_timestamp" in exported_data
        assert "metadata" in exported_data
        assert exported_data["metadata"]["total_entities"] == 2
        
        # 清空当前状态
        state_manager.current_states[session_id] = {}
        
        # 导入状态
        import_success = state_manager.import_state(
            session_id, original_state, merge_strategy="replace"
        )
        
        assert import_success
        assert state_manager.current_states[session_id] == original_state
        
        # 测试合并导入
        additional_state = {"user.789": {"name": "Charlie", "age": 35}}
        import_success = state_manager.import_state(
            session_id, additional_state, merge_strategy="merge"
        )
        
        assert import_success
        final_state = state_manager.current_states[session_id]
        assert len(final_state) == 3
        assert "user.789" in final_state