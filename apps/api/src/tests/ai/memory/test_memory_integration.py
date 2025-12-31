"""记忆系统集成测试"""

import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from unittest.mock import AsyncMock, Mock, patch
import json
import sys
import os
from src.ai.memory.models import Memory, MemoryType, MemoryStatus, MemoryFilters
from src.ai.memory.hierarchy_manager import MemoryHierarchyManager
from src.ai.memory.context_recall import ContextAwareRecall
from src.ai.memory.association_graph import MemoryAssociationGraph
from src.ai.memory.config import MemoryConfig
from src.services.memory_service import MemoryService

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

@pytest.mark.asyncio
class TestMemoryIntegration:
    """记忆系统集成测试"""
    
    @pytest.fixture
    async def memory_service(self):
        """创建记忆服务实例"""
        service = MemoryService()
        
        # 模拟存储层
        service.hierarchy_manager.storage = AsyncMock()
        service.hierarchy_manager.storage._initialized = True
        service.hierarchy_manager.storage.store_memory = AsyncMock(side_effect=lambda m: m)
        service.hierarchy_manager.storage.get_memory = AsyncMock()
        service.hierarchy_manager.storage.search_memories = AsyncMock(return_value=[])
        service.hierarchy_manager.storage.delete_memory = AsyncMock(return_value=True)
        service.hierarchy_manager.storage.bulk_insert = AsyncMock(return_value={
            "success_count": 0,
            "failed_count": 0,
            "errors": []
        })
        
        return service
        
    async def test_create_and_retrieve_memory(self, memory_service):
        """测试创建和检索记忆"""
        # 创建记忆
        memory = await memory_service.create_memory(
            content="测试记忆内容",
            memory_type=MemoryType.WORKING,
            session_id="test_session",
            importance=0.7,
            tags=["test"]
        )
        
        assert memory.content == "测试记忆内容"
        assert memory.type == MemoryType.WORKING
        assert memory.importance == 0.7
        assert "test" in memory.tags
        
        # 模拟获取记忆
        memory_service.hierarchy_manager.storage.get_memory.return_value = memory
        retrieved = await memory_service.get_memory(memory.id)
        
        assert retrieved == memory
        
    async def test_memory_hierarchy_promotion(self, memory_service):
        """测试记忆层级提升"""
        # 创建高重要性工作记忆
        working_memory = await memory_service.create_memory(
            content="重要的工作记忆",
            memory_type=MemoryType.WORKING,
            importance=0.8,
            session_id="test_session"
        )
        
        # 模拟访问增加
        working_memory.access_count = 5
        
        # 测试是否应该提升到情景记忆
        should_promote = memory_service.hierarchy_manager._should_promote_to_episodic(working_memory)
        assert should_promote
        
        # 创建情景记忆用于语义提升测试
        episodic_memory = Memory(
            type=MemoryType.EPISODIC,
            content="经常访问的重要知识",
            importance=0.9,
            access_count=10,
            created_at=utc_now() - timedelta(hours=2)
        )
        
        # 测试是否应该提升到语义记忆
        should_promote_semantic = memory_service.hierarchy_manager._should_promote_to_semantic(episodic_memory)
        assert should_promote_semantic
        
    async def test_memory_search_and_recall(self, memory_service):
        """测试记忆搜索和召回"""
        # 模拟搜索结果
        mock_memories = [
            Memory(
                id="mem1",
                type=MemoryType.SEMANTIC,
                content="关于Python编程的知识",
                importance=0.8,
                relevance_score=0.9
            ),
            Memory(
                id="mem2", 
                type=MemoryType.EPISODIC,
                content="用户学习Python的经历",
                importance=0.6,
                relevance_score=0.7
            )
        ]
        
        # 模拟上下文召回
        with patch.object(memory_service.context_recall, 'recall_relevant_memories') as mock_recall:
            mock_recall.return_value = [(m, m.relevance_score) for m in mock_memories]
            
            results = await memory_service.search_memories(
                query="Python编程学习",
                limit=10
            )
            
            assert len(results) == 2
            assert results[0][0].content == "关于Python编程的知识"
            assert results[0][1] == 0.9  # 相关性评分
            
    async def test_memory_association(self, memory_service):
        """测试记忆关联"""
        # 创建两个记忆
        memory1 = await memory_service.create_memory(
            content="Python基础语法",
            memory_type=MemoryType.SEMANTIC
        )
        memory2 = await memory_service.create_memory(
            content="Python高级特性",
            memory_type=MemoryType.SEMANTIC
        )
        
        # 关联记忆
        await memory_service.associate_memories(
            memory1.id,
            memory2.id,
            weight=0.8,
            association_type="related"
        )
        
        # 验证关联已添加
        assert memory2.id in memory1.related_memories
        
    async def test_memory_analytics(self, memory_service):
        """测试记忆分析"""
        # 模拟记忆数据
        mock_memories = [
            Memory(
                type=MemoryType.WORKING,
                content="工作记忆1",
                importance=0.5,
                access_count=3,
                created_at=utc_now()
            ),
            Memory(
                type=MemoryType.EPISODIC,
                content="情景记忆1", 
                importance=0.7,
                access_count=5,
                created_at=utc_now()
            ),
            Memory(
                type=MemoryType.SEMANTIC,
                content="语义记忆1",
                importance=0.9,
                access_count=10,
                created_at=utc_now()
            )
        ]
        
        memory_service.hierarchy_manager.storage.search_memories.return_value = mock_memories
        
        analytics = await memory_service.get_memory_analytics()
        
        assert analytics.total_memories == 3
        assert analytics.memories_by_type["working"] == 1
        assert analytics.memories_by_type["episodic"] == 1
        assert analytics.memories_by_type["semantic"] == 1
        assert analytics.avg_importance == (0.5 + 0.7 + 0.9) / 3
        
    async def test_memory_import_export(self, memory_service):
        """测试记忆导入导出"""
        # 测试导出
        mock_memories = [
            Memory(
                type=MemoryType.WORKING,
                content="导出测试记忆",
                importance=0.6
            )
        ]
        
        memory_service.hierarchy_manager.storage.search_memories.return_value = mock_memories
        
        exported_data = await memory_service.export_memories()
        assert len(exported_data) == 1
        assert exported_data[0]["content"] == "导出测试记忆"
        
        # 测试导入
        import_data = [
            {
                "type": "semantic",
                "content": "导入的语义记忆",
                "importance": 0.8,
                "tags": ["imported"]
            }
        ]
        
        memory_service.hierarchy_manager.storage.bulk_insert.return_value = {
            "success_count": 1,
            "failed_count": 0,
            "errors": []
        }
        
        import_result = await memory_service.import_memories(import_data)
        assert import_result.success_count == 1
        assert import_result.failed_count == 0
        
    async def test_memory_consolidation(self, memory_service):
        """测试记忆巩固"""
        # 添加工作记忆
        working_memories = [
            Memory(
                type=MemoryType.WORKING,
                content=f"工作记忆{i}",
                importance=0.6 + i * 0.1,
                access_count=i + 1
            ) for i in range(5)
        ]
        
        for memory in working_memories:
            memory_service.hierarchy_manager.working_memory.add(memory)
        
        # 模拟情景记忆检索
        episodic_memories = [
            Memory(
                type=MemoryType.EPISODIC,
                content="重要的情景记忆",
                importance=0.8,
                access_count=8,
                created_at=utc_now() - timedelta(hours=3)
            )
        ]
        
        memory_service.hierarchy_manager.episodic_memory.retrieve = AsyncMock(
            return_value=episodic_memories
        )
        
        # 模拟语义知识提取
        with patch.object(
            memory_service.hierarchy_manager.semantic_memory,
            'extract_knowledge'
        ) as mock_extract:
            mock_extract.return_value = Memory(
                type=MemoryType.SEMANTIC,
                content="提取的语义知识",
                importance=0.9
            )
            
            await memory_service.consolidate_memories("test_session")
            
            # 验证语义知识提取被调用
            mock_extract.assert_called_once()
            
    async def test_memory_cleanup(self, memory_service):
        """测试记忆清理"""
        # 模拟旧的低重要性记忆
        old_memories = [
            Memory(
                id=f"old_mem_{i}",
                type=MemoryType.WORKING,
                content=f"旧记忆{i}",
                importance=0.2,
                access_count=0,
                created_at=utc_now() - timedelta(days=35)
            ) for i in range(3)
        ]
        
        memory_service.hierarchy_manager.storage.search_memories.return_value = old_memories
        
        await memory_service.cleanup_old_memories(days_old=30, min_importance=0.3)
        
        # 验证删除被调用
        assert memory_service.hierarchy_manager.storage.delete_memory.call_count == 3

@pytest.mark.asyncio 
class TestContextAwareRecall:
    """上下文感知召回测试"""
    
    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        storage._initialized = True
        return storage
        
    @pytest.fixture
    def context_recall(self, mock_storage):
        config = MemoryConfig()
        return ContextAwareRecall(mock_storage, config)
        
    async def test_vector_search(self, context_recall):
        """测试向量搜索"""
        # 模拟嵌入生成
        with patch.object(context_recall, '_generate_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            # 模拟向量存储搜索
            context_recall.storage.vector_store = Mock()
            mock_search_result = [
                Mock(id="mem1", score=0.9),
                Mock(id="mem2", score=0.7)
            ]
            context_recall.storage.vector_store.search.return_value = mock_search_result
            
            # 模拟记忆检索
            mock_memories = [
                Memory(id="mem1", type=MemoryType.SEMANTIC, content="相关记忆1"),
                Memory(id="mem2", type=MemoryType.SEMANTIC, content="相关记忆2")
            ]
            context_recall.storage.get_memory.side_effect = lambda mid: next(
                (m for m in mock_memories if m.id == mid), None
            )
            
            results = await context_recall._vector_search([0.1] * 1536)
            
            assert len(results) == 2
            assert results[0][1] == 0.9  # 评分
            
    async def test_entity_extraction(self, context_recall):
        """测试实体提取"""
        # 模拟LLM实体提取
        with patch.object(context_recall.openai_client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"entities": ["Python", "编程", "学习"]}'
            mock_create.return_value = mock_response
            
            entities = await context_recall._extract_entities("学习Python编程")
            
            assert "Python" in entities
            assert "编程" in entities
            assert "学习" in entities
            
    async def test_recall_chain(self, context_recall):
        """测试链式召回"""
        start_memory = Memory(
            id="start",
            type=MemoryType.SEMANTIC,
            content="起始记忆",
            embedding=[0.1] * 1536
        )
        
        # 模拟相似记忆
        related_memories = [
            (Memory(id="rel1", type=MemoryType.SEMANTIC, content="相关1"), 0.8),
            (Memory(id="rel2", type=MemoryType.EPISODIC, content="相关2"), 0.6)
        ]
        
        with patch.object(context_recall, 'recall_by_similarity') as mock_similarity:
            mock_similarity.return_value = related_memories
            
            chain = await context_recall.recall_chain(start_memory, max_depth=2)
            
            assert len(chain) >= 1
            assert chain[0].id == "start"

@pytest.mark.asyncio
class TestMemoryAssociationGraph:
    """记忆关联图测试"""
    
    @pytest.fixture
    def mock_storage(self):
        return AsyncMock()
        
    @pytest.fixture
    def association_graph(self, mock_storage):
        return MemoryAssociationGraph(mock_storage)
        
    def test_add_memory_and_association(self, association_graph):
        """测试添加记忆和关联"""
        memory1 = Memory(
            id="mem1",
            type=MemoryType.SEMANTIC,
            content="记忆1",
            importance=0.8
        )
        memory2 = Memory(
            id="mem2", 
            type=MemoryType.SEMANTIC,
            content="记忆2",
            importance=0.6
        )
        
        # 添加节点
        association_graph.add_memory_node(memory1)
        association_graph.add_memory_node(memory2)
        
        assert "mem1" in association_graph.graph.nodes
        assert "mem2" in association_graph.graph.nodes
        
        # 添加关联
        association_graph.add_association(memory1, memory2, weight=0.7)
        
        assert association_graph.graph.has_edge("mem1", "mem2")
        assert association_graph.graph["mem1"]["mem2"]["weight"] == 0.7
        
    async def test_activate_related(self, association_graph):
        """测试激活相关记忆"""
        # 创建记忆网络
        memories = []
        for i in range(5):
            memory = Memory(
                id=f"mem{i}",
                type=MemoryType.SEMANTIC,
                content=f"记忆{i}",
                importance=0.5 + i * 0.1
            )
            memories.append(memory)
            association_graph.add_memory_node(memory)
            
        # 添加关联
        for i in range(4):
            association_graph.add_association(
                memories[i], 
                memories[i+1], 
                weight=0.6 + i * 0.1
            )
            
        # 模拟存储检索
        association_graph.storage.get_memory.side_effect = lambda mid: next(
            (m for m in memories if m.id == mid), None
        )
        
        # 激活相关记忆
        related = await association_graph.activate_related("mem0", depth=2)
        
        assert len(related) > 0
        # 验证激活强度递减
        if len(related) > 1:
            assert related[0][1] >= related[1][1]
            
    def test_find_memory_clusters(self, association_graph):
        """测试记忆聚类"""
        # 创建两个聚类
        cluster1_memories = [f"c1_mem{i}" for i in range(3)]
        cluster2_memories = [f"c2_mem{i}" for i in range(3)]
        
        # 添加节点
        for mid in cluster1_memories + cluster2_memories:
            memory = Memory(id=mid, type=MemoryType.SEMANTIC, content=f"内容{mid}")
            association_graph.add_memory_node(memory)
            
        # 聚类内连接
        for i in range(2):
            association_graph.graph.add_edge(cluster1_memories[i], cluster1_memories[i+1])
            association_graph.graph.add_edge(cluster2_memories[i], cluster2_memories[i+1])
            
        clusters = association_graph.find_memory_clusters(min_cluster_size=2)
        
        assert len(clusters) >= 2
        
    def test_graph_state_persistence(self, association_graph):
        """测试图状态持久化"""
        # 添加测试数据
        memory = Memory(id="test", type=MemoryType.SEMANTIC, content="测试")
        association_graph.add_memory_node(memory)
        association_graph._access_patterns["test"] = [utc_now()]
        
        # 保存状态
        state = association_graph.save_graph_state()
        
        assert "nodes" in state
        assert "edges" in state
        assert "access_patterns" in state
        
        # 创建新图并加载状态
        new_graph = MemoryAssociationGraph(AsyncMock())
        new_graph.load_graph_state(state)
        
        assert "test" in new_graph.graph.nodes
        assert "test" in new_graph._access_patterns
