"""
冲突解决器测试
"""

import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4
from ...sync.conflict_detector import (
    ConflictDetector, ConflictContext, ConflictSeverity, 
    ConflictCategory, ConflictDetectionResult
)
from ...sync.conflict_resolver import (
    ConflictResolver, ResolutionMethod, ResolutionStatus, 
    ResolutionResult, InteractionRequest
)
from ...sync.merge_strategies import MergeStrategies, MergeResult
from ...models.schemas.offline import (
    SyncOperation, SyncOperationType, VectorClock,
    ConflictType, ConflictResolutionStrategy

)

class TestConflictDetector:
    """冲突检测器测试"""
    
    @pytest.fixture
    def conflict_detector(self):
        """冲突检测器"""
        return ConflictDetector()
    
    def test_detect_update_update_conflict(self, conflict_detector):
        """测试检测更新-更新冲突"""
        session_id = "test_session"
        object_id = "test_object"
        
        # 创建本地操作
        local_op = SyncOperation(
            id="local_op",
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="users",
            object_id=object_id,
            object_type="user",
            data={"name": "Alice", "age": 31},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id="client1", clock={"client1": 2, "server": 1})
        )
        
        # 创建远程操作
        remote_op = SyncOperation(
            id="remote_op",
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="users",
            object_id=object_id,
            object_type="user",
            data={"name": "Alice Smith", "email": "alice@example.com"},
            client_timestamp=utc_now() - timedelta(minutes=1),
            vector_clock=VectorClock(node_id="client2", clock={"client2": 3, "server": 1})
        )
        
        # 检测冲突
        result = conflict_detector.detect_conflicts([local_op], [remote_op])
        
        # 验证检测结果
        assert isinstance(result, ConflictDetectionResult)
        assert result.has_conflicts
        assert len(result.conflicts) > 0
        
        conflict = result.conflicts[0]
        assert conflict.conflict_type == ConflictType.UPDATE_UPDATE
        assert conflict.local_operation.id == "local_op"
        assert conflict.remote_operation.id == "remote_op"
    
    def test_detect_create_create_conflict(self, conflict_detector):
        """测试检测创建-创建冲突"""
        session_id = "test_session"
        object_id = "new_object"
        
        # 创建两个并发的创建操作
        local_op = SyncOperation(
            id="local_create",
            session_id=session_id,
            operation_type=SyncOperationType.PUT,
            table_name="posts",
            object_id=object_id,
            object_type="post",
            data={"title": "My Post", "content": "Local content"},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id="client1", clock={"client1": 1})
        )
        
        remote_op = SyncOperation(
            id="remote_create",
            session_id=session_id,
            operation_type=SyncOperationType.PUT,
            table_name="posts",
            object_id=object_id,
            object_type="post",
            data={"title": "My Post", "content": "Remote content"},
            client_timestamp=utc_now() - timedelta(seconds=30),
            vector_clock=VectorClock(node_id="client2", clock={"client2": 1})
        )
        
        result = conflict_detector.detect_conflicts([local_op], [remote_op])
        
        assert result.has_conflicts
        assert len(result.conflicts) > 0
        conflict = result.conflicts[0]
        assert conflict.conflict_type == ConflictType.CREATE_CREATE
    
    def test_detect_update_delete_conflict(self, conflict_detector):
        """测试检测更新-删除冲突"""
        session_id = "test_session"
        object_id = "target_object"
        
        # 本地更新操作
        local_op = SyncOperation(
            id="local_update",
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="documents",
            object_id=object_id,
            object_type="document",
            data={"status": "published"},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id="client1", clock={"client1": 3, "server": 2})
        )
        
        # 远程删除操作
        remote_op = SyncOperation(
            id="remote_delete",
            session_id=session_id,
            operation_type=SyncOperationType.DELETE,
            table_name="documents",
            object_id=object_id,
            object_type="document",
            data={},
            client_timestamp=utc_now() - timedelta(minutes=2),
            vector_clock=VectorClock(node_id="client2", clock={"client2": 2, "server": 2})
        )
        
        result = conflict_detector.detect_conflicts([local_op], [remote_op])
        
        assert result.has_conflicts
        conflict = result.conflicts[0]
        assert conflict.conflict_type == ConflictType.UPDATE_DELETE
    
    def test_no_conflict_sequential_operations(self, conflict_detector):
        """测试顺序操作无冲突"""
        session_id = "test_session"
        object_id = "sequential_object"
        
        # 顺序操作（非并发）
        local_op = SyncOperation(
            id="local_seq",
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="items",
            object_id=object_id,
            object_type="item",
            data={"value": 100},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id="client1", clock={"client1": 2, "server": 1})
        )
        
        remote_op = SyncOperation(
            id="remote_seq",
            session_id=session_id,
            operation_type=SyncOperationType.PATCH,
            table_name="items",
            object_id=object_id,
            object_type="item",
            data={"status": "active"},
            client_timestamp=utc_now() - timedelta(minutes=5),
            vector_clock=VectorClock(node_id="client2", clock={"client1": 1, "client2": 1, "server": 1})
        )
        
        result = conflict_detector.detect_conflicts([local_op], [remote_op])
        
        # 应该没有冲突，因为操作是顺序的
        assert not result.has_conflicts
    
    def test_conflict_summary_generation(self, conflict_detector):
        """测试冲突摘要生成"""
        # 创建多个冲突上下文
        conflicts = []
        
        for i in range(3):
            context = ConflictContext(
                local_operation=SyncOperation(
                    id=f"local_{i}",
                    session_id="test",
                    operation_type=SyncOperationType.PATCH,
                    table_name="test_table",
                    object_id=f"obj_{i}",
                    object_type="test",
                    data={"value": i},
                    client_timestamp=utc_now(),
                    vector_clock=VectorClock(node_id="client1")
                ),
                remote_operation=SyncOperation(
                    id=f"remote_{i}",
                    session_id="test",
                    operation_type=SyncOperationType.PATCH,
                    table_name="test_table",
                    object_id=f"obj_{i}",
                    object_type="test",
                    data={"value": i + 10},
                    client_timestamp=utc_now(),
                    vector_clock=VectorClock(node_id="client2")
                ),
                conflict_type=ConflictType.UPDATE_UPDATE,
                conflict_category=ConflictCategory.DATA_CONFLICT,
                severity=ConflictSeverity.MEDIUM,
                auto_resolvable=i < 2,  # 前两个可自动解决
                confidence_score=0.8
            )
            conflicts.append(context)
        
        summary = conflict_detector._generate_conflict_summary(conflicts)
        
        assert summary["total_conflicts"] == 3
        assert summary["auto_resolvable"] == 2
        assert summary["manual_resolution_required"] == 1
        assert "severity_distribution" in summary
        assert "category_distribution" in summary
        assert "type_distribution" in summary

class TestConflictResolver:
    """冲突解决器测试"""
    
    @pytest.fixture
    def conflict_resolver(self):
        """冲突解决器"""
        return ConflictResolver()
    
    @pytest.fixture
    def sample_conflict_context(self):
        """示例冲突上下文"""
        local_op = SyncOperation(
            id="local_test",
            session_id="test_session",
            operation_type=SyncOperationType.PATCH,
            table_name="users",
            object_id="user_123",
            object_type="user",
            data={"name": "Alice Johnson", "age": 31},
            client_timestamp=utc_now(),
            vector_clock=VectorClock(node_id="client1", clock={"client1": 2})
        )
        
        remote_op = SyncOperation(
            id="remote_test",
            session_id="test_session",
            operation_type=SyncOperationType.PATCH,
            table_name="users",
            object_id="user_123",
            object_type="user",
            data={"name": "Alice Smith", "email": "alice@example.com"},
            client_timestamp=utc_now() - timedelta(minutes=1),
            vector_clock=VectorClock(node_id="client2", clock={"client2": 3})
        )
        
        return ConflictContext(
            local_operation=local_op,
            remote_operation=remote_op,
            conflict_type=ConflictType.UPDATE_UPDATE,
            conflict_category=ConflictCategory.DATA_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            auto_resolvable=True,
            confidence_score=0.8
        )
    
    @pytest.mark.asyncio
    async def test_resolve_last_writer_wins(self, conflict_resolver, sample_conflict_context):
        """测试最后写入者获胜策略"""
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.LAST_WRITER_WINS
        )
        
        assert isinstance(result, ResolutionResult)
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.LAST_WRITER_WINS
        assert result.resolved_data is not None
        
        # 应该选择时间戳更新的数据
        local_timestamp = sample_conflict_context.local_operation.client_timestamp
        remote_timestamp = sample_conflict_context.remote_operation.client_timestamp
        
        if local_timestamp >= remote_timestamp:
            expected_data = sample_conflict_context.local_operation.data
        else:
            expected_data = sample_conflict_context.remote_operation.data
        
        assert result.resolved_data == expected_data
    
    @pytest.mark.asyncio
    async def test_resolve_first_writer_wins(self, conflict_resolver, sample_conflict_context):
        """测试第一写入者获胜策略"""
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.FIRST_WRITER_WINS
        )
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.FIRST_WRITER_WINS
        
        # 应该选择时间戳更早的数据
        local_timestamp = sample_conflict_context.local_operation.client_timestamp
        remote_timestamp = sample_conflict_context.remote_operation.client_timestamp
        
        if local_timestamp <= remote_timestamp:
            expected_data = sample_conflict_context.local_operation.data
        else:
            expected_data = sample_conflict_context.remote_operation.data
        
        assert result.resolved_data == expected_data
    
    @pytest.mark.asyncio
    async def test_resolve_client_wins(self, conflict_resolver, sample_conflict_context):
        """测试客户端获胜策略"""
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.CLIENT_WINS
        )
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.CLIENT_WINS
        assert result.resolved_data == sample_conflict_context.local_operation.data
    
    @pytest.mark.asyncio
    async def test_resolve_server_wins(self, conflict_resolver, sample_conflict_context):
        """测试服务器获胜策略"""
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.SERVER_WINS
        )
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.SERVER_WINS
        assert result.resolved_data == sample_conflict_context.remote_operation.data
    
    @pytest.mark.asyncio
    async def test_resolve_merge_strategy(self, conflict_resolver, sample_conflict_context):
        """测试合并策略"""
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.MERGE
        )
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.MERGE
        assert result.resolved_data is not None
        
        # 验证合并结果包含两边的数据
        resolved_data = result.resolved_data
        local_data = sample_conflict_context.local_operation.data
        remote_data = sample_conflict_context.remote_operation.data
        
        # 应该包含两边独有的字段
        assert "age" in resolved_data  # 来自本地
        assert "email" in resolved_data  # 来自远程
    
    @pytest.mark.asyncio
    async def test_resolve_manual_strategy(self, conflict_resolver, sample_conflict_context):
        """测试手动解决策略"""
        custom_data = {"name": "Alice Custom", "age": 32, "email": "alice.custom@example.com"}
        
        result = await conflict_resolver.resolve_conflict(
            sample_conflict_context,
            preferred_strategy=ConflictResolutionStrategy.MANUAL,
            user_context={"resolved_data": custom_data}
        )
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_strategy == ConflictResolutionStrategy.MANUAL
        assert result.resolved_data == custom_data
    
    @pytest.mark.asyncio
    async def test_automatic_resolution_strategy_selection(self, conflict_resolver):
        """测试自动解决策略选择"""
        # 创建一个可自动解决的冲突
        auto_resolvable_context = ConflictContext(
            local_operation=SyncOperation(
                id="auto_local",
                session_id="auto_test",
                operation_type=SyncOperationType.PATCH,
                table_name="settings",
                object_id="setting_1",
                object_type="setting",
                data={"value": "local_value"},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id="client1")
            ),
            remote_operation=SyncOperation(
                id="auto_remote",
                session_id="auto_test",
                operation_type=SyncOperationType.PATCH,
                table_name="settings",
                object_id="setting_1",
                object_type="setting",
                data={"value": "remote_value"},
                client_timestamp=utc_now() - timedelta(minutes=2),
                vector_clock=VectorClock(node_id="client2")
            ),
            conflict_type=ConflictType.UPDATE_UPDATE,
            conflict_category=ConflictCategory.DATA_CONFLICT,
            severity=ConflictSeverity.LOW,
            auto_resolvable=True,
            confidence_score=0.9
        )
        
        result = await conflict_resolver.resolve_conflict(auto_resolvable_context)
        
        assert result.status == ResolutionStatus.RESOLVED
        assert result.resolution_method == ResolutionMethod.AUTOMATIC
    
    @pytest.mark.asyncio
    async def test_user_interaction_request(self, conflict_resolver):
        """测试用户交互请求"""
        # 创建一个需要手动干预的冲突
        manual_context = ConflictContext(
            local_operation=SyncOperation(
                id="manual_local",
                session_id="manual_test",
                operation_type=SyncOperationType.DELETE,
                table_name="documents",
                object_id="doc_1",
                object_type="document",
                data={},
                client_timestamp=utc_now(),
                vector_clock=VectorClock(node_id="client1")
            ),
            remote_operation=SyncOperation(
                id="manual_remote",
                session_id="manual_test",
                operation_type=SyncOperationType.PATCH,
                table_name="documents",
                object_id="doc_1",
                object_type="document",
                data={"content": "updated content"},
                client_timestamp=utc_now() - timedelta(minutes=1),
                vector_clock=VectorClock(node_id="client2")
            ),
            conflict_type=ConflictType.DELETE_UPDATE,
            conflict_category=ConflictCategory.DATA_CONFLICT,
            severity=ConflictSeverity.HIGH,
            auto_resolvable=False,
            confidence_score=0.2
        )
        
        # 禁用自动解决
        conflict_resolver.resolution_config["auto_resolve_enabled"] = False
        
        result = await conflict_resolver.resolve_conflict(manual_context)
        
        assert result.status == ResolutionStatus.REQUIRES_MANUAL_INTERVENTION
        assert result.resolution_method == ResolutionMethod.INTERACTIVE
        assert "interaction_request_id" in result.metadata
        
        # 检查是否创建了待处理的交互
        assert len(conflict_resolver.get_pending_interactions()) > 0
    
    @pytest.mark.asyncio
    async def test_handle_user_response(self, conflict_resolver, sample_conflict_context):
        """测试处理用户响应"""
        # 先创建一个交互请求
        conflict_resolver.resolution_config["auto_resolve_enabled"] = False
        result = await conflict_resolver.resolve_conflict(sample_conflict_context)
        
        assert result.status == ResolutionStatus.REQUIRES_MANUAL_INTERVENTION
        request_id = result.metadata["interaction_request_id"]
        
        # 模拟用户选择客户端获胜策略
        user_response = await conflict_resolver.handle_user_response(
            request_id,
            ConflictResolutionStrategy.CLIENT_WINS
        )
        
        assert user_response.status == ResolutionStatus.RESOLVED
        assert user_response.resolution_strategy == ConflictResolutionStrategy.CLIENT_WINS
        assert user_response.resolution_method == ResolutionMethod.INTERACTIVE
        
        # 交互应该被清理
        assert len(conflict_resolver.get_pending_interactions()) == 0
    
    def test_resolution_statistics(self, conflict_resolver):
        """测试解决统计信息"""
        # 模拟一些统计数据
        conflict_resolver.resolution_stats["total_resolutions"] = 10
        conflict_resolver.resolution_stats["automatic_resolutions"] = 6
        conflict_resolver.resolution_stats["interactive_resolutions"] = 3
        conflict_resolver.resolution_stats["manual_resolutions"] = 1
        conflict_resolver.resolution_stats["total_resolution_time_ms"] = 5000.0
        
        stats = conflict_resolver.get_resolution_statistics()
        
        assert stats["total_resolutions"] == 10
        assert stats["automatic_resolution_rate"] == 0.6
        assert stats["interactive_resolution_rate"] == 0.3
        assert stats["manual_resolution_rate"] == 0.1
        assert stats["average_resolution_time_ms"] == 500.0

class TestMergeStrategies:
    """合并策略测试"""
    
    @pytest.fixture
    def merge_strategies(self):
        """合并策略"""
        return MergeStrategies()
    
    def test_three_way_merge_no_conflicts(self, merge_strategies):
        """测试无冲突的三路合并"""
        base_data = {"name": "Alice", "age": 30}
        local_data = {"name": "Alice", "age": 31, "email": "alice@example.com"}  # 添加email，修改age
        remote_data = {"name": "Alice", "age": 30, "city": "New York"}  # 添加city
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # 应该包含所有更改
        assert merged_data["name"] == "Alice"
        assert merged_data["age"] == 31  # 本地修改
        assert merged_data["email"] == "alice@example.com"  # 本地添加
        assert merged_data["city"] == "New York"  # 远程添加
        assert confidence > 0.5
    
    def test_three_way_merge_with_conflicts(self, merge_strategies):
        """测试有冲突的三路合并"""
        base_data = {"name": "Alice", "status": "active"}
        local_data = {"name": "Alice Johnson", "status": "active"}  # 修改name
        remote_data = {"name": "Alice Smith", "status": "active"}   # 也修改name
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # 应该有合并结果，但置信度较低
        assert "name" in merged_data
        assert merged_data["status"] == "active"
        assert confidence < 0.8  # 由于冲突，置信度应该较低
    
    def test_merge_list_fields(self, merge_strategies):
        """测试列表字段合并"""
        base_data = {"tags": ["python", "programming"]}
        local_data = {"tags": ["python", "programming", "web"]}  # 添加web
        remote_data = {"tags": ["python", "programming", "ai"]}  # 添加ai
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # 列表应该被合并（对于可合并的字段类型）
        if "tags" in merge_strategies.auto_mergeable_types:
            assert set(merged_data["tags"]) >= {"python", "programming", "web", "ai"}
        else:
            # 如果不能自动合并，应该有结果但可能有冲突
            assert "tags" in merged_data
    
    def test_merge_timestamp_fields(self, merge_strategies):
        """测试时间戳字段合并"""
        base_time = "2023-01-01T12:00:00Z"
        local_time = "2023-01-01T12:30:00Z"
        remote_time = "2023-01-01T12:15:00Z"
        
        base_data = {"created_at": base_time, "updated_at": base_time}
        local_data = {"created_at": base_time, "updated_at": local_time}
        remote_data = {"created_at": base_time, "updated_at": remote_time}
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # created_at 应该保持不变
        assert merged_data["created_at"] == base_time
        
        # updated_at 应该取最新的
        assert merged_data["updated_at"] == local_time  # local_time 更新
    
    def test_merge_dict_fields(self, merge_strategies):
        """测试字典字段合并"""
        base_data = {"config": {"theme": "light", "lang": "en"}}
        local_data = {"config": {"theme": "dark", "lang": "en", "timezone": "UTC"}}
        remote_data = {"config": {"theme": "light", "lang": "zh", "region": "Asia"}}
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # 字典应该递归合并
        config = merged_data["config"]
        assert "timezone" in config  # 本地添加
        assert "region" in config    # 远程添加
        assert config["lang"] == "zh"  # 远程修改（如果只有远程修改）
    
    def test_merge_accumulative_fields(self, merge_strategies):
        """测试累积字段合并"""
        base_data = {"views": 100, "likes": 10}
        local_data = {"views": 150, "likes": 12}  # 增加50 views, 2 likes
        remote_data = {"views": 120, "likes": 15}  # 增加20 views, 5 likes
        
        merged_data, confidence = merge_strategies.three_way_merge(
            base_data, local_data, remote_data
        )
        
        # 累积字段应该累加增量
        # local_delta: views +50, likes +2
        # remote_delta: views +20, likes +5
        # expected: base + local_delta + remote_delta
        expected_views = 100 + 50 + 20  # 170
        expected_likes = 10 + 2 + 5     # 17
        
        assert merged_data["views"] == expected_views
        assert merged_data["likes"] == expected_likes
    
    def test_semantic_merge(self, merge_strategies):
        """测试语义合并"""
        local_data = {"status": "published", "priority": "high"}
        remote_data = {"status": "draft", "priority": "medium"}
        
        # 定义schema信息
        schema_info = {
            "status": {
                "type": "enum",
                "constraints": {
                    "values": ["draft", "review", "published"],
                    "priority_order": ["draft", "review", "published"]
                }
            },
            "priority": {
                "type": "enum",
                "constraints": {
                    "values": ["low", "medium", "high"],
                    "priority_order": ["high", "medium", "low"]
                }
            }
        }
        
        result = merge_strategies.semantic_merge(local_data, remote_data, schema_info)
        
        assert isinstance(result, MergeResult)
        assert result.merged_data["status"] in ["draft", "review", "published"]
        assert result.merged_data["priority"] in ["low", "medium", "high"]
    
    def test_custom_merge_rules(self, merge_strategies):
        """测试自定义合并规则"""
        local_data = {"field1": "local", "field2": "local", "field3": "local"}
        remote_data = {"field1": "remote", "field2": "remote", "field3": "remote"}
        
        # 定义自定义规则
        merge_rules = {
            "field1": {"type": "always_local"},
            "field2": {"type": "always_remote"},
            # field3 使用默认规则
        }
        
        result = merge_strategies.custom_merge(local_data, remote_data, merge_rules)
        
        assert isinstance(result, MergeResult)
        assert result.merged_data["field1"] == "local"
        assert result.merged_data["field2"] == "remote"
        assert "field3" in result.merged_data
    
    def test_string_intelligent_merge(self, merge_strategies):
        """测试智能字符串合并"""
        # 测试包含关系
        result1 = merge_strategies._merge_string_intelligently("Hello", "Hello World")
        assert result1 == "Hello World"
        
        # 测试大小写差异
        result2 = merge_strategies._merge_string_intelligently("hello world", "Hello World")
        assert result2 == "Hello World"  # 保留更多大写字母的版本
        
        # 测试空格差异
        result3 = merge_strategies._merge_string_intelligently("hello world", "hello  world")
        assert result3 == "hello  world"  # 保留较长的版本
        
        # 测试无法合并的情况
        result4 = merge_strategies._merge_string_intelligently("completely", "different")
        assert result4 is None
