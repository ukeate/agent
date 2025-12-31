"""记忆系统单元测试"""

import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from unittest.mock import Mock, AsyncMock, patch
import uuid
import sys
import os
from src.ai.memory.models import (
    Memory, MemoryType, MemoryStatus,
    MemoryCreateRequest, MemoryFilters
)
from src.ai.memory.storage import MemoryStorage
from src.ai.memory.hierarchy_manager import (
    MemoryHierarchyManager,
    WorkingMemoryBuffer,
    EpisodicMemoryStore,
    SemanticMemoryStore
)
from src.ai.memory.config import MemoryConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class TestMemoryModels:
    """测试记忆数据模型"""
    
    def test_memory_creation(self):
        """测试记忆创建"""
        memory = Memory(
            type=MemoryType.WORKING,
            content="测试记忆内容",
            importance=0.8,
            tags=["test", "unit"]
        )
        
        assert memory.id is not None
        assert memory.type == MemoryType.WORKING
        assert memory.content == "测试记忆内容"
        assert memory.importance == 0.8
        assert memory.access_count == 0
        assert memory.status == MemoryStatus.ACTIVE
        assert "test" in memory.tags
        
    def test_memory_create_request(self):
        """测试创建记忆请求"""
        request = MemoryCreateRequest(
            type=MemoryType.EPISODIC,
            content="事件记忆",
            metadata={"event": "test"},
            importance=0.7
        )
        
        assert request.type == MemoryType.EPISODIC
        assert request.content == "事件记忆"
        assert request.metadata["event"] == "test"
        
    def test_memory_filters(self):
        """测试记忆过滤器"""
        filters = MemoryFilters(
            memory_types=[MemoryType.SEMANTIC],
            min_importance=0.5,
            tags=["knowledge"]
        )
        
        assert MemoryType.SEMANTIC in filters.memory_types
        assert filters.min_importance == 0.5
        assert "knowledge" in filters.tags

class TestWorkingMemoryBuffer:
    """测试工作记忆缓冲区"""
    
    def test_buffer_add_and_get(self):
        """测试添加和获取"""
        buffer = WorkingMemoryBuffer(capacity=3)
        
        memory1 = Memory(
            id="mem1",
            type=MemoryType.WORKING,
            content="记忆1"
        )
        memory2 = Memory(
            id="mem2",
            type=MemoryType.WORKING,
            content="记忆2"
        )
        
        buffer.add(memory1)
        buffer.add(memory2)
        
        assert buffer.get("mem1") == memory1
        assert buffer.get("mem2") == memory2
        assert len(buffer.get_all()) == 2
        
    def test_buffer_capacity_limit(self):
        """测试容量限制"""
        buffer = WorkingMemoryBuffer(capacity=2)
        
        memories = [
            Memory(id=f"mem{i}", type=MemoryType.WORKING, content=f"记忆{i}")
            for i in range(3)
        ]
        
        for memory in memories:
            buffer.add(memory)
            
        # 应该只保留最后2个
        all_memories = buffer.get_all()
        assert len(all_memories) == 2
        assert buffer.get("mem0") is None  # 第一个被移除
        assert buffer.get("mem1") is not None
        assert buffer.get("mem2") is not None
        
    def test_buffer_is_full(self):
        """测试缓冲区满状态"""
        buffer = WorkingMemoryBuffer(capacity=2)
        
        assert not buffer.is_full()
        
        buffer.add(Memory(type=MemoryType.WORKING, content="1"))
        assert not buffer.is_full()
        
        buffer.add(Memory(type=MemoryType.WORKING, content="2"))
        assert buffer.is_full()
        
    def test_buffer_clear(self):
        """测试清空缓冲区"""
        buffer = WorkingMemoryBuffer(capacity=3)
        
        for i in range(3):
            buffer.add(Memory(
                id=f"mem{i}",
                type=MemoryType.WORKING,
                content=f"记忆{i}"
            ))
            
        buffer.clear()
        assert len(buffer.get_all()) == 0
        assert buffer.get("mem0") is None

@pytest.mark.asyncio
class TestMemoryStorage:
    """测试记忆存储层"""
    
    @pytest.fixture
    async def mock_storage(self):
        """创建模拟存储"""
        config = MemoryConfig()
        storage = MemoryStorage(config)
        
        # 模拟数据库连接
        storage.vector_store = Mock()
        storage.redis_cache = AsyncMock()
        storage.postgres_pool = AsyncMock()
        storage._initialized = True
        
        return storage
        
    async def test_store_memory(self, mock_storage):
        """测试存储记忆"""
        memory = Memory(
            type=MemoryType.EPISODIC,
            content="测试事件",
            embedding=[0.1] * 1536
        )
        
        # 模拟数据库操作
        mock_storage.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        mock_storage.vector_store.upsert = Mock()
        
        result = await mock_storage.store_memory(memory)
        
        assert result == memory
        mock_storage.vector_store.upsert.assert_called_once()
        
    async def test_get_memory(self, mock_storage):
        """测试获取记忆"""
        memory_id = "test-id"
        
        # 模拟缓存未命中
        mock_storage.redis_cache.get.return_value = None
        
        # 模拟数据库查询
        mock_row = {
            'id': memory_id,
            'type': 'episodic',
            'content': '测试内容',
            'metadata': '{}',
            'importance': 0.5,
            'access_count': 0,
            'created_at': utc_now(),
            'last_accessed': utc_now(),
            'decay_factor': 0.5,
            'status': 'active',
            'session_id': None,
            'user_id': None,
            'related_memories': [],
            'tags': [],
            'source': None
        }
        
        mock_storage.postgres_pool.acquire.return_value.__aenter__.return_value.fetchrow = AsyncMock(
            return_value=mock_row
        )
        mock_storage.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        result = await mock_storage.get_memory(memory_id)
        
        assert result is not None
        assert result.id == memory_id
        assert result.type == MemoryType.EPISODIC
        
    async def test_search_memories(self, mock_storage):
        """测试搜索记忆"""
        filters = MemoryFilters(
            memory_types=[MemoryType.SEMANTIC],
            min_importance=0.5
        )
        
        # 模拟数据库查询结果
        mock_rows = [
            {
                'id': f'mem-{i}',
                'type': 'semantic',
                'content': f'知识{i}',
                'metadata': '{}',
                'importance': 0.6 + i * 0.1,
                'access_count': i,
                'created_at': utc_now(),
                'last_accessed': utc_now(),
                'decay_factor': 0.5,
                'status': 'active',
                'session_id': None,
                'user_id': None,
                'related_memories': [],
                'tags': [],
                'source': None
            }
            for i in range(3)
        ]
        
        mock_storage.postgres_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=mock_rows
        )
        
        results = await mock_storage.search_memories(filters, limit=10)
        
        assert len(results) == 3
        assert all(m.type == MemoryType.SEMANTIC for m in results)
        assert all(m.importance >= 0.5 for m in results)

@pytest.mark.asyncio
class TestMemoryHierarchyManager:
    """测试记忆层级管理器"""
    
    @pytest.fixture
    async def mock_manager(self):
        """创建模拟管理器"""
        config = MemoryConfig()
        manager = MemoryHierarchyManager(config)
        
        # 模拟存储
        manager.storage = AsyncMock()
        manager.storage.store_memory = AsyncMock(side_effect=lambda m: m)
        manager.storage.search_memories = AsyncMock(return_value=[])
        
        return manager
        
    async def test_add_working_memory(self, mock_manager):
        """测试添加工作记忆"""
        memory = await mock_manager.add_memory(
            content="工作记忆内容",
            memory_type=MemoryType.WORKING,
            importance=0.5
        )
        
        assert memory.type == MemoryType.WORKING
        assert memory.content == "工作记忆内容"
        assert len(mock_manager.working_memory.get_all()) == 1
        
    async def test_add_episodic_memory(self, mock_manager):
        """测试添加情景记忆"""
        memory = await mock_manager.add_memory(
            content="事件描述",
            memory_type=MemoryType.EPISODIC,
            importance=0.7
        )
        
        assert memory.type == MemoryType.EPISODIC
        mock_manager.storage.store_memory.assert_called_once()
        
    @patch('ai.memory.hierarchy_manager.get_openai_client')
    async def test_add_semantic_memory(self, mock_openai, mock_manager):
        """测试添加语义记忆"""
        # 模拟嵌入生成
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create = Mock(
            return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
        )
        
        memory = await mock_manager.add_memory(
            content="知识点",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8
        )
        
        assert memory.type == MemoryType.SEMANTIC
        mock_manager.storage.store_memory.assert_called_once()
        
    def test_should_promote_to_episodic(self, mock_manager):
        """测试判断是否应提升到情景记忆"""
        # 重要性高的记忆
        memory1 = Memory(
            type=MemoryType.WORKING,
            content="重要",
            importance=0.7
        )
        assert mock_manager._should_promote_to_episodic(memory1)
        
        # 访问次数多的记忆
        memory2 = Memory(
            type=MemoryType.WORKING,
            content="频繁",
            importance=0.4,
            access_count=4
        )
        assert mock_manager._should_promote_to_episodic(memory2)
        
        # 不重要且访问少的记忆
        memory3 = Memory(
            type=MemoryType.WORKING,
            content="普通",
            importance=0.3,
            access_count=1
        )
        assert not mock_manager._should_promote_to_episodic(memory3)
        
    def test_should_promote_to_semantic(self, mock_manager):
        """测试判断是否应提升到语义记忆"""
        # 满足所有条件
        memory1 = Memory(
            type=MemoryType.EPISODIC,
            content="重要知识",
            importance=0.8,
            access_count=6,
            created_at=utc_now() - timedelta(hours=2)
        )
        assert mock_manager._should_promote_to_semantic(memory1)
        
        # 重要性不够
        memory2 = Memory(
            type=MemoryType.EPISODIC,
            content="一般知识",
            importance=0.6,
            access_count=6,
            created_at=utc_now() - timedelta(hours=2)
        )
        assert not mock_manager._should_promote_to_semantic(memory2)
        
        # 时间太短
        memory3 = Memory(
            type=MemoryType.EPISODIC,
            content="新知识",
            importance=0.8,
            access_count=6,
            created_at=utc_now() - timedelta(minutes=30)
        )
        assert not mock_manager._should_promote_to_semantic(memory3)
        
    async def test_get_relevant_memories(self, mock_manager):
        """测试获取相关记忆"""
        # 添加不同类型的记忆
        working_mem = Memory(
            type=MemoryType.WORKING,
            content="工作",
            importance=0.5
        )
        mock_manager.working_memory.add(working_mem)
        
        episodic_mems = [
            Memory(type=MemoryType.EPISODIC, content=f"事件{i}", importance=0.6 + i * 0.1)
            for i in range(3)
        ]
        mock_manager.episodic_memory.retrieve = AsyncMock(return_value=episodic_mems)
        
        semantic_mems = [
            Memory(type=MemoryType.SEMANTIC, content=f"知识{i}", importance=0.7 + i * 0.1)
            for i in range(2)
        ]
        mock_manager.semantic_memory.retrieve = AsyncMock(return_value=semantic_mems)
        
        # 获取所有类型
        results = await mock_manager.get_relevant_memories(
            context="测试上下文",
            limit=10
        )
        
        assert len(results) == 6  # 1 working + 3 episodic + 2 semantic
        
        # 按重要性排序
        importances = [m.importance for m in results]
        assert importances == sorted(importances, reverse=True)
