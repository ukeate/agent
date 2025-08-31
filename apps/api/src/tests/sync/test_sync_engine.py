"""
同步引擎测试
"""

import pytest
import tempfile
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4

from ...sync.sync_engine import (
    SyncEngine, SyncTask, SyncPriority, SyncDirection, 
    SyncStatus, SyncResult
)
from ...sync.vector_clock import VectorClockManager, CausalRelation
from ...sync.delta_calculator import DeltaCalculator, DeltaType
from ...offline.models import OfflineDatabase
from ...models.schemas.offline import (
    SyncOperation, SyncOperationType, VectorClock, 
    NetworkStatus, ConflictType
)


class TestSyncEngine:
    """同步引擎测试"""
    
    @pytest.fixture
    def temp_database(self):
        """临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            database = OfflineDatabase(temp_file.name)
            yield database
            database.close()
    
    @pytest.fixture
    def sync_engine(self, temp_database):
        """同步引擎"""
        return SyncEngine(temp_database)
    
    @pytest.mark.asyncio
    async def test_create_sync_task(self, sync_engine):
        """测试创建同步任务"""
        session_id = "test_session"
        
        # 添加一些待同步操作
        operations = []
        for i in range(5):
            operation = SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PUT,
                table_name="test_table",
                object_id=f"obj_{i}",
                object_type="test_object",
                data={"value": i},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id=session_id)
            )
            sync_engine.database.add_operation(operation)
            operations.append(operation)
        
        # 创建上传任务
        task_id = await sync_engine.create_sync_task(
            session_id=session_id,
            direction=SyncDirection.UPLOAD,
            priority=SyncPriority.HIGH
        )
        
        assert task_id is not None
        
        # 检查任务状态
        task_status = sync_engine.get_sync_task_status(task_id)
        assert task_status is not None
        assert task_status["session_id"] == session_id
        assert task_status["direction"] == SyncDirection.UPLOAD.value
        assert task_status["priority"] == SyncPriority.HIGH.value
        assert task_status["status"] == SyncStatus.PENDING.value
        assert task_status["total_operations"] == len(operations)
    
    @pytest.mark.asyncio
    async def test_execute_upload_task(self, sync_engine):
        """测试执行上传任务"""
        session_id = "upload_test"
        
        # 添加待同步操作
        operations = []
        for i in range(3):
            operation = SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PUT,
                table_name="upload_table",
                object_id=f"upload_obj_{i}",
                object_type="upload_object",
                data={"name": f"Object {i}", "value": i * 10},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id=session_id)
            )
            sync_engine.database.add_operation(operation)
            operations.append(operation)
        
        # 创建并执行任务
        task_id = await sync_engine.create_sync_task(
            session_id=session_id,
            direction=SyncDirection.UPLOAD,
            priority=SyncPriority.NORMAL
        )
        
        result = await sync_engine.execute_sync_task(task_id)
        
        # 验证结果
        assert isinstance(result, SyncResult)
        assert result.task_id == task_id
        assert result.total_operations == len(operations)
        assert result.successful_operations >= 0
        assert result.failed_operations >= 0
        assert result.successful_operations + result.failed_operations == result.total_operations
        assert result.duration_seconds > 0
        assert result.throughput_ops_per_second >= 0
    
    @pytest.mark.asyncio
    async def test_execute_download_task(self, sync_engine):
        """测试执行下载任务"""
        session_id = "download_test"
        
        # 创建下载任务
        task_id = await sync_engine.create_sync_task(
            session_id=session_id,
            direction=SyncDirection.DOWNLOAD,
            priority=SyncPriority.HIGH
        )
        
        result = await sync_engine.execute_sync_task(task_id)
        
        # 验证结果
        assert isinstance(result, SyncResult)
        assert result.task_id == task_id
        assert result.total_operations >= 0  # 可能有服务器操作
        assert result.duration_seconds > 0
    
    @pytest.mark.asyncio
    async def test_execute_bidirectional_task(self, sync_engine):
        """测试执行双向同步任务"""
        session_id = "bidirectional_test"
        
        # 添加本地操作
        operation = SyncOperation(
            id=str(uuid4()),
            session_id=session_id,
            operation_type=SyncOperationType.PUT,
            table_name="bidirectional_table",
            object_id="bidirectional_obj",
            object_type="bidirectional_object",
            data={"status": "local_update"},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id=session_id)
        )
        sync_engine.database.add_operation(operation)
        
        # 创建双向同步任务
        task_id = await sync_engine.create_sync_task(
            session_id=session_id,
            direction=SyncDirection.BIDIRECTIONAL,
            priority=SyncPriority.CRITICAL
        )
        
        result = await sync_engine.execute_sync_task(task_id)
        
        # 验证结果
        assert isinstance(result, SyncResult)
        assert result.task_id == task_id
        assert result.total_operations >= 1  # 至少有本地操作
    
    def test_task_priority_sorting(self, sync_engine):
        """测试任务优先级排序"""
        session_id = "priority_test"
        
        # 创建不同优先级的任务
        priorities = [
            SyncPriority.LOW,
            SyncPriority.CRITICAL,
            SyncPriority.NORMAL,
            SyncPriority.HIGH,
            SyncPriority.BACKGROUND
        ]
        
        tasks = []
        for priority in priorities:
            task = SyncTask(
                id=str(uuid4()),
                session_id=session_id,
                direction=SyncDirection.UPLOAD,
                priority=priority,
                operation_ids=[]
            )
            sync_engine.task_queue.append(task)
            tasks.append(task)
        
        # 排序任务队列
        sync_engine._sort_task_queue()
        
        # 验证排序结果（数值越小优先级越高）
        sorted_priorities = [task.priority for task in sync_engine.task_queue]
        expected_order = [
            SyncPriority.CRITICAL,
            SyncPriority.HIGH,
            SyncPriority.NORMAL,
            SyncPriority.LOW,
            SyncPriority.BACKGROUND
        ]
        
        assert sorted_priorities == expected_order
    
    @pytest.mark.asyncio
    async def test_task_control_operations(self, sync_engine):
        """测试任务控制操作"""
        session_id = "control_test"
        
        # 创建任务
        task_id = await sync_engine.create_sync_task(
            session_id=session_id,
            direction=SyncDirection.UPLOAD,
            priority=SyncPriority.NORMAL
        )
        
        # 测试暂停
        paused = await sync_engine.pause_sync_task(task_id)
        assert paused
        
        task_status = sync_engine.get_sync_task_status(task_id)
        assert task_status["status"] == SyncStatus.PAUSED.value
        
        # 测试恢复
        resumed = await sync_engine.resume_sync_task(task_id)
        assert resumed
        
        task_status = sync_engine.get_sync_task_status(task_id)
        assert task_status["status"] == SyncStatus.PENDING.value
        
        # 测试取消
        cancelled = await sync_engine.cancel_sync_task(task_id)
        assert cancelled
        
        # 取消的任务应该从队列中移除
        task_status = sync_engine.get_sync_task_status(task_id)
        assert task_status is None or task_status["status"] == SyncStatus.CANCELLED.value
    
    def test_operation_filtering(self, sync_engine):
        """测试操作过滤"""
        # 创建测试操作
        operations = [
            SyncOperation(
                id="op1",
                session_id="test",
                operation_type=SyncOperationType.PUT,
                table_name="users",
                object_id="user1",
                object_type="user",
                data={},
                client_timestamp=utc_now() - timedelta(hours=2),
                vector_clock=VectorClock(node_id="test")
            ),
            SyncOperation(
                id="op2",
                session_id="test",
                operation_type=SyncOperationType.PATCH,
                table_name="posts",
                object_id="post1",
                object_type="post",
                data={},
                client_timestamp=utc_now() - timedelta(hours=1),
                vector_clock=VectorClock(node_id="test")
            ),
            SyncOperation(
                id="op3",
                session_id="test",
                operation_type=SyncOperationType.DELETE,
                table_name="users",
                object_id="user2",
                object_type="user",
                data={},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id="test")
            )
        ]
        
        # 测试按表名过滤
        filter_criteria = {"table_names": ["users"]}
        filtered = sync_engine._filter_operations(operations, filter_criteria)
        assert len(filtered) == 2
        assert all(op.table_name == "users" for op in filtered)
        
        # 测试按操作类型过滤
        filter_criteria = {"operation_types": [SyncOperationType.PUT, SyncOperationType.DELETE]}
        filtered = sync_engine._filter_operations(operations, filter_criteria)
        assert len(filtered) == 2
        assert all(op.operation_type in [SyncOperationType.PUT, SyncOperationType.DELETE] for op in filtered)
        
        # 测试按时间过滤
        from_time = utc_now() - timedelta(hours=1, minutes=30)
        filter_criteria = {"from_time": from_time}
        filtered = sync_engine._filter_operations(operations, filter_criteria)
        assert len(filtered) == 2
    
    def test_list_sync_tasks(self, sync_engine):
        """测试列出同步任务"""
        session_id = "list_test"
        
        # 创建多个任务
        task_ids = []
        for i in range(3):
            task = SyncTask(
                id=str(uuid4()),
                session_id=session_id,
                direction=SyncDirection.UPLOAD,
                priority=SyncPriority.NORMAL,
                operation_ids=[],
                status=SyncStatus.PENDING if i < 2 else SyncStatus.COMPLETED
            )
            sync_engine.task_queue.append(task)
            task_ids.append(task.id)
        
        # 列出所有任务
        all_tasks = sync_engine.list_sync_tasks()
        assert len(all_tasks) == 3
        
        # 按会话过滤
        session_tasks = sync_engine.list_sync_tasks(session_id=session_id)
        assert len(session_tasks) == 3
        
        # 按状态过滤
        pending_tasks = sync_engine.list_sync_tasks(
            status_filter=[SyncStatus.PENDING]
        )
        assert len(pending_tasks) == 2
        
        completed_tasks = sync_engine.list_sync_tasks(
            status_filter=[SyncStatus.COMPLETED]
        )
        assert len(completed_tasks) == 1
    
    def test_sync_statistics(self, sync_engine):
        """测试同步统计"""
        # 模拟一些统计数据
        sync_engine.total_synced_operations = 100
        sync_engine.total_failed_operations = 10
        sync_engine.total_conflicts_resolved = 5
        sync_engine.last_sync_time = utc_now()
        
        # 添加一些任务到队列
        for i in range(3):
            task = SyncTask(
                id=str(uuid4()),
                session_id="stats_test",
                direction=SyncDirection.UPLOAD,
                priority=SyncPriority.NORMAL,
                operation_ids=[],
                status=SyncStatus.PENDING
            )
            sync_engine.task_queue.append(task)
        
        # 获取统计信息
        stats = sync_engine.get_sync_statistics()
        
        # 验证统计信息
        assert stats["total_synced_operations"] == 100
        assert stats["total_failed_operations"] == 10
        assert stats["total_conflicts_resolved"] == 5
        assert stats["queued_tasks"] == 3
        assert stats["sync_efficiency"] == 100 / 110  # 成功率


class TestVectorClockManager:
    """向量时钟管理器测试"""
    
    @pytest.fixture
    def clock_manager(self):
        """向量时钟管理器"""
        return VectorClockManager()
    
    def test_create_and_increment_clock(self, clock_manager):
        """测试创建和递增时钟"""
        node_id = "node1"
        
        # 获取或创建时钟
        clock = clock_manager.get_or_create_clock(node_id)
        assert clock.node_id == node_id
        assert clock.clock.get(node_id, 0) == 0
        
        # 递增时钟
        incremented_clock = clock_manager.increment_clock(node_id)
        assert incremented_clock.clock[node_id] == 1
        
        # 再次递增
        incremented_clock = clock_manager.increment_clock(node_id)
        assert incremented_clock.clock[node_id] == 2
    
    def test_compare_clocks(self, clock_manager):
        """测试时钟比较"""
        # 创建两个时钟
        clock1 = VectorClock(node_id="node1")
        clock1.clock = {"node1": 2, "node2": 1}
        
        clock2 = VectorClock(node_id="node2")
        clock2.clock = {"node1": 1, "node2": 2}
        
        # 并发关系
        relation = clock_manager.compare_clocks(clock1, clock2)
        assert relation == CausalRelation.CONCURRENT
        
        # Before关系
        clock3 = VectorClock(node_id="node1")
        clock3.clock = {"node1": 1, "node2": 1}
        
        clock4 = VectorClock(node_id="node2")
        clock4.clock = {"node1": 2, "node2": 2}
        
        relation = clock_manager.compare_clocks(clock3, clock4)
        assert relation == CausalRelation.BEFORE
        
        # After关系
        relation = clock_manager.compare_clocks(clock4, clock3)
        assert relation == CausalRelation.AFTER
        
        # Equal关系
        clock5 = VectorClock(node_id="node1")
        clock5.clock = {"node1": 1, "node2": 1}
        
        clock6 = VectorClock(node_id="node2")
        clock6.clock = {"node1": 1, "node2": 1}
        
        relation = clock_manager.compare_clocks(clock5, clock6)
        assert relation == CausalRelation.EQUAL
    
    def test_detect_conflict(self, clock_manager):
        """测试冲突检测"""
        # 并发时钟（冲突）
        clock1 = VectorClock(node_id="node1")
        clock1.clock = {"node1": 2, "node2": 1}
        
        clock2 = VectorClock(node_id="node2")
        clock2.clock = {"node1": 1, "node2": 2}
        
        assert clock_manager.detect_conflict(clock1, clock2) == True
        
        # 有序时钟（无冲突）
        clock3 = VectorClock(node_id="node1")
        clock3.clock = {"node1": 1, "node2": 1}
        
        clock4 = VectorClock(node_id="node2")
        clock4.clock = {"node1": 2, "node2": 2}
        
        assert clock_manager.detect_conflict(clock3, clock4) == False
    
    def test_merge_clocks(self, clock_manager):
        """测试时钟合并"""
        local_clock = VectorClock(node_id="local")
        local_clock.clock = {"local": 3, "remote": 1}
        
        remote_clock = VectorClock(node_id="remote")
        remote_clock.clock = {"local": 2, "remote": 4}
        
        # 合并时钟
        result = clock_manager.merge_clocks(local_clock, remote_clock)
        
        # 验证合并结果
        assert result.merged_clock.clock["local"] == 4  # max(3, 2) + 1（本地递增）
        assert result.merged_clock.clock["remote"] == 4  # max(1, 4)
        assert result.conflicts_detected == False  # 这种情况不是冲突
    
    def test_sync_with_multiple_remotes(self, clock_manager):
        """测试与多个远程时钟同步"""
        local_node = "local"
        
        # 创建多个远程时钟
        remote_clocks = []
        for i in range(3):
            clock = VectorClock(node_id=f"remote{i}")
            clock.clock = {f"remote{i}": i + 1, "local": i}
            remote_clocks.append(clock)
        
        # 同步
        results = clock_manager.sync_with_remote(local_node, remote_clocks)
        
        # 验证结果
        assert len(results) == 3
        for i, result in enumerate(results.values()):
            assert isinstance(result.merged_clock, VectorClock)
            assert result.merged_clock.node_id == local_node
    
    def test_causally_ready_check(self, clock_manager):
        """测试因果就绪检查"""
        local_clock = VectorClock(node_id="local")
        local_clock.clock = {"local": 2, "remote": 3}
        
        # 因果就绪的事件
        ready_event = VectorClock(node_id="remote")
        ready_event.clock = {"local": 2, "remote": 4}  # remote递增1，其他不变
        
        assert clock_manager.is_causally_ready(ready_event, local_clock) == True
        
        # 因果未就绪的事件
        not_ready_event = VectorClock(node_id="remote")
        not_ready_event.clock = {"local": 3, "remote": 4}  # local超前了
        
        assert clock_manager.is_causally_ready(not_ready_event, local_clock) == False


class TestDeltaCalculator:
    """增量计算器测试"""
    
    @pytest.fixture
    def delta_calculator(self):
        """增量计算器"""
        return DeltaCalculator()
    
    def test_calculate_object_delta_creation(self, delta_calculator):
        """测试对象创建差异"""
        object_id = "obj_123"
        table_name = "users"
        old_data = None
        new_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, old_data, new_data
        )
        
        assert delta.object_id == object_id
        assert delta.table_name == table_name
        assert len(delta.operations) == 1
        assert delta.operations[0].operation_type == DeltaType.OBJECT_CREATION
        assert delta.operations[0].new_value == new_data
        assert delta.checksum is not None
    
    def test_calculate_object_delta_deletion(self, delta_calculator):
        """测试对象删除差异"""
        object_id = "obj_456"
        table_name = "users"
        old_data = {"name": "Bob", "age": 25}
        new_data = None
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, old_data, new_data
        )
        
        assert len(delta.operations) == 1
        assert delta.operations[0].operation_type == DeltaType.OBJECT_DELETION
        assert delta.operations[0].old_value == old_data
    
    def test_calculate_field_deltas(self, delta_calculator):
        """测试字段差异计算"""
        old_data = {
            "name": "Alice",
            "age": 30,
            "address": {"city": "New York", "zip": "10001"}
        }
        new_data = {
            "name": "Alice Smith",  # 修改
            "age": 30,              # 不变
            "email": "alice@example.com",  # 添加
            "address": {"city": "New York", "zip": "10002", "country": "USA"}  # 嵌套修改和添加
        }
        
        object_id = "field_test"
        table_name = "users"
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, old_data, new_data
        )
        
        # 验证操作类型
        operation_types = [op.operation_type for op in delta.operations]
        assert DeltaType.FIELD_MODIFICATION in operation_types  # name修改
        assert DeltaType.FIELD_ADDITION in operation_types      # email添加
        
        # 查找具体操作
        name_ops = [op for op in delta.operations if op.path == "name"]
        assert len(name_ops) == 1
        assert name_ops[0].operation_type == DeltaType.FIELD_MODIFICATION
        assert name_ops[0].old_value == "Alice"
        assert name_ops[0].new_value == "Alice Smith"
        
        email_ops = [op for op in delta.operations if op.path == "email"]
        assert len(email_ops) == 1
        assert email_ops[0].operation_type == DeltaType.FIELD_ADDITION
        assert email_ops[0].new_value == "alice@example.com"
    
    def test_calculate_list_deltas(self, delta_calculator):
        """测试列表差异计算"""
        old_data = {"items": [1, 2, 3]}
        new_data = {"items": [1, 5, 3, 4]}  # 修改第二个元素，添加第四个元素
        
        object_id = "list_test"
        table_name = "test"
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, old_data, new_data
        )
        
        # 查找列表相关操作
        list_ops = [op for op in delta.operations if op.path.startswith("items")]
        assert len(list_ops) >= 2  # 至少有修改和插入操作
        
        # 检查修改操作
        modification_ops = [op for op in list_ops if op.operation_type == DeltaType.LIST_MODIFICATION]
        assert len(modification_ops) >= 1
        
        # 检查插入操作
        insertion_ops = [op for op in list_ops if op.operation_type == DeltaType.LIST_INSERTION]
        assert len(insertion_ops) >= 1
    
    def test_apply_delta_to_object(self, delta_calculator):
        """测试将差异应用到对象"""
        original_data = {"name": "Alice", "age": 30}
        
        # 创建一个修改差异
        object_id = "apply_test"
        table_name = "users"
        new_data = {"name": "Alice Smith", "age": 31, "email": "alice@example.com"}
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, original_data, new_data
        )
        
        # 应用差异
        result = delta_calculator.apply_delta_to_object(original_data, delta)
        
        # 验证结果
        assert result["name"] == "Alice Smith"
        assert result["age"] == 31
        assert result["email"] == "alice@example.com"
    
    def test_path_parsing(self, delta_calculator):
        """测试路径解析"""
        # 测试简单路径
        path1 = "name"
        keys1 = delta_calculator._parse_path(path1)
        assert keys1 == ["name"]
        
        # 测试嵌套路径
        path2 = "address.city"
        keys2 = delta_calculator._parse_path(path2)
        assert keys2 == ["address", "city"]
        
        # 测试数组路径
        path3 = "items[0]"
        keys3 = delta_calculator._parse_path(path3)
        assert keys3 == ["items", 0]
        
        # 测试复杂路径
        path4 = "users[0].address.coordinates[1]"
        keys4 = delta_calculator._parse_path(path4)
        assert keys4 == ["users", 0, "address", "coordinates", 1]
    
    def test_delta_compression(self, delta_calculator):
        """测试差异压缩"""
        # 创建一个大的差异对象
        large_data = {}
        for i in range(100):
            large_data[f"field_{i}"] = f"value_{i}" * 100  # 创建大字符串
        
        object_id = "compression_test"
        table_name = "large_table"
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, None, large_data
        )
        
        # 压缩差异
        compressed_delta = delta_calculator.compress_object_delta(delta)
        
        # 验证压缩
        assert compressed_delta.compressed_size <= compressed_delta.original_size
        assert compressed_delta.compression_algorithm != delta_calculator.CompressionAlgorithm.NONE
    
    def test_estimate_sync_size(self, delta_calculator):
        """测试同步大小估算"""
        # 创建测试操作
        operations = []
        for i in range(10):
            operation = SyncOperation(
                id=str(uuid4()),
                session_id="size_test",
                operation_type=SyncOperationType.PUT,
                table_name="test_table",
                object_id=f"obj_{i}",
                object_type="test_object",
                data={"value": i, "description": f"Object {i}" * 50},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id="test")
            )
            operations.append(operation)
        
        # 估算大小
        size_estimate = delta_calculator.estimate_sync_size(operations)
        
        # 验证估算结果
        assert size_estimate["total_operations"] == 10
        assert size_estimate["total_size_bytes"] > 0
        assert size_estimate["estimated_compressed_size"] < size_estimate["total_size_bytes"]
        assert size_estimate["average_operation_size"] > 0
        assert size_estimate["max_operation_size"] >= size_estimate["min_operation_size"]
    
    def test_validate_delta_integrity(self, delta_calculator):
        """测试差异完整性验证"""
        object_id = "integrity_test"
        table_name = "test"
        old_data = {"value": 1}
        new_data = {"value": 2}
        
        delta = delta_calculator.calculate_object_delta(
            object_id, table_name, old_data, new_data
        )
        
        # 验证完整性（应该通过）
        assert delta_calculator.validate_delta_integrity(delta) == True
        
        # 修改差异破坏完整性
        delta.operations[0].new_value = {"value": 999}
        
        # 验证完整性（应该失败）
        assert delta_calculator.validate_delta_integrity(delta) == False