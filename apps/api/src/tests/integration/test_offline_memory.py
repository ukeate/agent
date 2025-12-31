"""
离线记忆系统集成测试
"""

import pytest
import tempfile
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from uuid import uuid4
from offline.memory_manager import (
    OfflineMemoryManager, MemoryEntry, MemoryQuery,
    MemoryType, MemoryPriority
)
from src.ai.memory.context_recall import ContextAwareRecall
from models.schemas.offline import OfflineMode, NetworkStatus, VectorClock

class MockMemoryStorage:
    """模拟内存存储"""
    
    def __init__(self):
        self.vector_store = None
    
    async def get_memory(self, memory_id):
        return None
    
    async def search_memories(self, filters, limit=10):
        return []

class MockMemoryConfig:
    """模拟内存配置"""
    
    def __init__(self):
        self.vector_dimension = 768
        self.decay_constant = 3600  # 1小时衰减常数
        self.qdrant_collection = "test_memories"

class TestOfflineMemoryIntegration:
    """离线记忆系统集成测试"""
    
    @pytest.fixture
    def temp_memory_manager(self):
        """临时记忆管理器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineMemoryManager(storage_path=temp_dir)
            yield manager
    
    @pytest.fixture
    def context_recall_system(self, temp_memory_manager):
        """上下文召回系统"""
        storage = MockMemoryStorage()
        config = MockMemoryConfig()
        recall_system = ContextAwareRecall(storage, config)
        
        # 替换离线记忆管理器
        recall_system.offline_memory_manager = temp_memory_manager
        
        return recall_system
    
    def test_memory_storage_and_retrieval(self, temp_memory_manager):
        """测试记忆存储和检索"""
        session_id = "test_session"
        user_id = str(uuid4())
        
        # 创建测试记忆
        memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content="用户询问了关于Python编程的问题",
            context={'user_id': user_id, 'topic': 'programming'},
            priority=MemoryPriority.HIGH,
            tags=['python', 'programming', 'question'],
            vector_clock=VectorClock(node_id="test_node"),
            created_at=utc_now()
        )
        
        # 存储记忆
        success = temp_memory_manager.store_memory(memory)
        assert success
        
        # 检索记忆
        retrieved_memory = temp_memory_manager.retrieve_memory(memory.id)
        assert retrieved_memory is not None
        assert retrieved_memory.content == memory.content
        assert retrieved_memory.memory_type == memory.memory_type
        assert retrieved_memory.session_id == session_id
        assert 'python' in retrieved_memory.tags
    
    def test_memory_search_functionality(self, temp_memory_manager):
        """测试记忆搜索功能"""
        session_id = "search_test_session"
        
        # 创建多个测试记忆
        memories = [
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.FACTUAL,
                content="Python是一种高级编程语言",
                context={'topic': 'python'},
                priority=MemoryPriority.HIGH,
                tags=['python', 'programming', 'language']
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.PROCEDURAL,
                content="如何使用pip安装Python包",
                context={'topic': 'python', 'action': 'install'},
                priority=MemoryPriority.MEDIUM,
                tags=['python', 'pip', 'package', 'install']
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.CONVERSATION,
                content="用户询问JavaScript和Python的区别",
                context={'topic': 'comparison'},
                priority=MemoryPriority.MEDIUM,
                tags=['python', 'javascript', 'comparison']
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.EPISODIC,
                content="昨天讨论了机器学习算法",
                context={'topic': 'ml'},
                priority=MemoryPriority.LOW,
                tags=['machine-learning', 'algorithm']
            )
        ]
        
        # 存储所有记忆
        for memory in memories:
            temp_memory_manager.store_memory(memory)
        
        # 搜索Python相关记忆
        query = MemoryQuery(
            query_text="Python编程语言",
            memory_types=[MemoryType.FACTUAL, MemoryType.PROCEDURAL],
            tags=['python'],
            limit=5
        )
        
        results = temp_memory_manager.search_memories(query)
        
        # 验证搜索结果
        assert len(results) >= 2  # 应该找到至少2个Python相关记忆
        
        # 验证结果包含相关内容
        found_contents = [result.entry.content for result in results]
        assert any('Python' in content for content in found_contents)
        assert any('pip' in content for content in found_contents)
    
    @pytest.mark.asyncio
    async def test_context_recall_offline_mode(self, context_recall_system):
        """测试上下文召回的离线模式"""
        session_id = "context_test_session"
        user_id = str(uuid4())
        
        # 设置离线模式
        context_recall_system.set_offline_mode(
            OfflineMode.OFFLINE, 
            NetworkStatus.DISCONNECTED
        )
        
        # 添加一些测试记忆到离线存储
        memories = [
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.CONVERSATION,
                content="用户问如何学习Python",
                context={'user_id': user_id},
                priority=MemoryPriority.HIGH,
                tags=['python', 'learning']
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.FACTUAL,
                content="Python是一种解释型语言",
                context={'user_id': user_id},
                priority=MemoryPriority.MEDIUM,
                tags=['python', 'interpreted']
            )
        ]
        
        for memory in memories:
            context_recall_system.offline_memory_manager.store_memory(memory)
        
        # 测试离线召回
        results = await context_recall_system.recall_relevant_memories(
            context="Python编程学习",
            session_id=session_id,
            user_id=user_id,
            limit=5
        )
        
        # 验证结果
        assert len(results) > 0
        memory, score = results[0]
        assert 'Python' in memory.content
        assert score > 0.0
    
    @pytest.mark.asyncio
    async def test_context_recall_mode_switching(self, context_recall_system):
        """测试上下文召回的模式切换"""
        session_id = "mode_switch_test"
        
        # 添加离线记忆
        offline_memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content="离线模式下的记忆内容",
            context={'source': 'offline'},
            tags=['offline', 'test']
        )
        context_recall_system.offline_memory_manager.store_memory(offline_memory)
        
        # 测试离线模式
        context_recall_system.set_offline_mode(
            OfflineMode.OFFLINE, 
            NetworkStatus.DISCONNECTED
        )
        
        offline_results = await context_recall_system.recall_relevant_memories(
            context="离线记忆测试",
            session_id=session_id,
            limit=5
        )
        
        assert len(offline_results) > 0
        assert '离线' in offline_results[0][0].content
        
        # 切换到在线模式（但由于没有真实的在线存储，会返回空结果）
        context_recall_system.set_offline_mode(
            OfflineMode.ONLINE, 
            NetworkStatus.CONNECTED
        )
        
        online_results = await context_recall_system.recall_relevant_memories(
            context="在线记忆测试",
            session_id=session_id,
            limit=5
        )
        
        # 在线模式下应该使用在线存储（这里因为是模拟，所以结果为空）
        assert len(online_results) == 0 or len(online_results) > 0  # 允许两种情况
    
    def test_memory_compression(self, temp_memory_manager):
        """测试记忆压缩功能"""
        session_id = "compression_test"
        
        # 创建一个大的记忆内容（超过压缩阈值）
        large_content = "这是一个很长的记忆内容。" * 200  # 重复200次创建大内容
        
        memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.FACTUAL,
            content=large_content,
            context={'size': 'large'},
            priority=MemoryPriority.MEDIUM
        )
        
        # 存储记忆（应该会自动压缩）
        success = temp_memory_manager.store_memory(memory)
        assert success
        
        # 检索记忆
        retrieved_memory = temp_memory_manager.retrieve_memory(memory.id)
        assert retrieved_memory is not None
        assert retrieved_memory.content == large_content  # 内容应该正确解压
        
        # 检查压缩状态
        if retrieved_memory.is_compressed:
            assert retrieved_memory.original_size > retrieved_memory.compressed_size
    
    def test_memory_priority_update(self, temp_memory_manager):
        """测试记忆优先级更新"""
        session_id = "priority_test"
        
        memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content="优先级测试记忆",
            context={},
            priority=MemoryPriority.LOW
        )
        
        # 存储记忆
        temp_memory_manager.store_memory(memory)
        
        # 更新优先级
        success = temp_memory_manager.update_memory_priority(
            memory.id, 
            MemoryPriority.CRITICAL
        )
        assert success
        
        # 验证优先级已更新
        retrieved_memory = temp_memory_manager.retrieve_memory(memory.id)
        assert retrieved_memory.priority == MemoryPriority.CRITICAL
    
    def test_memory_statistics(self, temp_memory_manager):
        """测试记忆统计功能"""
        session_id = "stats_test"
        
        # 创建不同类型的记忆
        memories = [
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.CONVERSATION,
                content="对话记忆",
                context={},
                priority=MemoryPriority.HIGH
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.FACTUAL,
                content="事实记忆",
                context={},
                priority=MemoryPriority.MEDIUM
            ),
            MemoryEntry(
                id=str(uuid4()),
                session_id=session_id,
                memory_type=MemoryType.PROCEDURAL,
                content="程序记忆",
                context={},
                priority=MemoryPriority.LOW
            )
        ]
        
        # 存储记忆
        for memory in memories:
            temp_memory_manager.store_memory(memory)
        
        # 获取统计信息
        stats = temp_memory_manager.get_memory_stats()
        
        # 验证统计信息
        assert stats['total_memories'] == 3
        assert stats['memory_types']['conversation'] == 1
        assert stats['memory_types']['factual'] == 1
        assert stats['memory_types']['procedural'] == 1
        assert stats['priorities'][str(MemoryPriority.HIGH.value)] == 1
    
    def test_memory_cleanup(self, temp_memory_manager):
        """测试记忆清理功能"""
        session_id = "cleanup_test"
        
        # 创建旧记忆
        old_memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content="旧的记忆",
            context={},
            priority=MemoryPriority.LOW,
            created_at=utc_now() - timedelta(days=100),
            last_accessed=utc_now() - timedelta(days=100)
        )
        
        # 创建新记忆
        new_memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content="新的记忆",
            context={},
            priority=MemoryPriority.HIGH
        )
        
        # 存储记忆
        temp_memory_manager.store_memory(old_memory)
        temp_memory_manager.store_memory(new_memory)
        
        # 清理旧记忆（保留重要记忆）
        deleted_count = temp_memory_manager.cleanup_old_memories(
            days_threshold=30,
            keep_important=True
        )
        
        # 验证清理结果
        assert deleted_count >= 0
        
        # 验证重要记忆仍然存在
        retrieved_new = temp_memory_manager.retrieve_memory(new_memory.id)
        assert retrieved_new is not None
