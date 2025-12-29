"""记忆管理服务层"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
from src.ai.memory.models import (
    Memory, MemoryType, MemoryStatus,
    MemoryCreateRequest, MemoryUpdateRequest,
    MemoryQuery, MemoryFilters, MemoryAnalytics,
    ImportResult
)
from src.ai.memory.hierarchy_manager import MemoryHierarchyManager
from src.ai.memory.context_recall import ContextAwareRecall
from src.ai.memory.association_graph import MemoryAssociationGraph
from src.ai.memory.config import MemoryConfig

logger = get_logger(__name__)

class MemoryService:
    """记忆管理服务"""
    
    def __init__(self):
        self.config = MemoryConfig()
        self.hierarchy_manager = MemoryHierarchyManager(self.config)
        self.context_recall = ContextAwareRecall(
            self.hierarchy_manager.storage,
            self.config
        )
        self.association_graph = MemoryAssociationGraph(
            self.hierarchy_manager.storage
        )
        self._initialized = False
        
    async def initialize(self):
        """初始化服务"""
        if not self._initialized:
            await self.hierarchy_manager.initialize()
            self._initialized = True
            
    async def create_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> Memory:
        """创建新记忆"""
        await self.initialize()
        
        memory = await self.hierarchy_manager.add_memory(
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            importance=importance,
            tags=tags or [],
            source=source
        )
        
        # 添加到关联图
        self.association_graph.add_memory_node(memory)
        
        logger.info(f"创建记忆: {memory.id} ({memory_type.value if hasattr(memory_type, 'value') else str(memory_type)})")
        return memory
        
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        await self.initialize()
        return await self.hierarchy_manager.storage.get_memory(memory_id)
        
    async def update_memory(
        self,
        memory_id: str,
        update_request: MemoryUpdateRequest
    ) -> Optional[Memory]:
        """更新记忆"""
        await self.initialize()
        
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
            
        # 更新字段
        if update_request.content is not None:
            memory.content = update_request.content
        if update_request.metadata is not None:
            memory.metadata.update(update_request.metadata)
        if update_request.importance is not None:
            memory.importance = update_request.importance
        if update_request.tags is not None:
            memory.tags = update_request.tags
        if update_request.status is not None:
            memory.status = update_request.status
            
        # 保存更新
        await self.hierarchy_manager.storage.store_memory(memory)
        
        logger.info(f"更新记忆: {memory_id}")
        return memory
        
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        await self.initialize()
        return await self.hierarchy_manager.storage.delete_memory(memory_id)
        
    async def search_memories(
        self,
        query: str,
        filters: Optional[MemoryFilters] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """搜索记忆"""
        await self.initialize()
        
        # 使用上下文召回进行搜索
        results = await self.context_recall.recall_relevant_memories(
            context=query,
            session_id=filters.session_id if filters else None,
            user_id=filters.user_id if filters else None,
            memory_types=filters.memory_types if filters else None,
            limit=limit
        )
        
        return results
        
    async def get_session_memories(
        self,
        session_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[Memory]:
        """获取会话的所有记忆"""
        await self.initialize()
        return await self.hierarchy_manager.storage.get_session_memories(
            session_id,
            memory_type
        )
        
    async def get_memory_analytics(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> MemoryAnalytics:
        """获取记忆分析统计"""
        await self.initialize()
        
        # 构建过滤器
        filters = MemoryFilters(
            session_id=session_id,
            user_id=user_id,
            created_after=start_time,
            created_before=end_time
        )
        
        # 获取记忆
        memories = await self.hierarchy_manager.storage.search_memories(
            filters,
            limit=10000
        )
        
        # 统计分析
        total_memories = len(memories)
        memories_by_type = {}
        memories_by_status = {}
        total_importance = 0
        total_access_count = 0
        
        for memory in memories:
            # 按类型统计
            type_key = memory.type.value
            memories_by_type[type_key] = memories_by_type.get(type_key, 0) + 1
            
            # 按状态统计
            status_key = memory.status.value
            memories_by_status[status_key] = memories_by_status.get(status_key, 0) + 1
            
            # 累计重要性和访问次数
            total_importance += memory.importance
            total_access_count += memory.access_count
            
        # 计算平均值
        avg_importance = total_importance / total_memories if total_memories > 0 else 0
        avg_access_count = total_access_count / total_memories if total_memories > 0 else 0
        
        # 获取最频繁访问的记忆
        memories.sort(key=lambda m: m.access_count, reverse=True)
        most_accessed = memories[:10]
        
        # 获取最近的记忆
        memories.sort(key=lambda m: m.created_at, reverse=True)
        recent_memories = memories[:10]
        
        # 计算增长率(简化版)
        if start_time and end_time:
            time_range = (end_time - start_time).days
            growth_rate = total_memories / max(time_range, 1)
        else:
            growth_rate = 0
            
        # 估算存储使用量
        total_bytes = 0
        for memory in memories:
            try:
                total_bytes += len((memory.content or "").encode("utf-8"))
                total_bytes += len(json.dumps(memory.metadata or {}, ensure_ascii=False).encode("utf-8"))
            except Exception:
                continue
        storage_usage_mb = total_bytes / (1024 * 1024)
        
        from src.ai.memory.models import MemoryResponse
        
        return MemoryAnalytics(
            total_memories=total_memories,
            memories_by_type=memories_by_type,
            memories_by_status=memories_by_status,
            avg_importance=avg_importance,
            total_access_count=total_access_count,
            avg_access_count=avg_access_count,
            most_accessed_memories=[
                MemoryResponse.from_memory(m) for m in most_accessed
            ],
            recent_memories=[
                MemoryResponse.from_memory(m) for m in recent_memories
            ],
            memory_growth_rate=growth_rate,
            storage_usage_mb=storage_usage_mb
        )
        
    async def import_memories(
        self,
        memories_data: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> ImportResult:
        """导入记忆"""
        await self.initialize()
        
        memories = []
        errors = []
        
        for data in memories_data:
            try:
                memory = Memory(**data)
                if session_id:
                    memory.session_id = session_id
                memories.append(memory)
            except Exception as e:
                errors.append(f"解析记忆失败: {str(e)}")
                
        # 批量插入
        result = await self.hierarchy_manager.storage.bulk_insert(memories)
        
        return ImportResult(
            success_count=result["success_count"],
            failed_count=result["failed_count"],
            errors=errors + result["errors"],
            imported_ids=[m.id for m in memories[:result["success_count"]]]
        )
        
    async def export_memories(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Dict[str, Any]]:
        """导出记忆"""
        await self.initialize()
        
        filters = MemoryFilters(
            session_id=session_id,
            user_id=user_id,
            memory_types=memory_types
        )
        
        memories = await self.hierarchy_manager.storage.search_memories(
            filters,
            limit=10000
        )
        
        return [memory.model_dump(mode="json") for memory in memories]
        
    async def associate_memories(
        self,
        memory1_id: str,
        memory2_id: str,
        weight: float = 0.5,
        association_type: str = "related"
    ):
        """关联两个记忆"""
        await self.initialize()
        
        memory1 = await self.get_memory(memory1_id)
        memory2 = await self.get_memory(memory2_id)
        
        if memory1 and memory2:
            self.association_graph.add_association(
                memory1,
                memory2,
                weight,
                association_type
            )
            
            # 更新记忆的关联列表
            if memory2_id not in memory1.related_memories:
                memory1.related_memories.append(memory2_id)
                await self.hierarchy_manager.storage.store_memory(memory1)
                
            if memory1_id not in memory2.related_memories:
                memory2.related_memories.append(memory1_id)
                await self.hierarchy_manager.storage.store_memory(memory2)
                
    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 2,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """获取相关记忆"""
        await self.initialize()
        return await self.association_graph.activate_related(
            memory_id,
            depth,
            min_weight=0.3
        )
        
    async def consolidate_memories(self, session_id: str):
        """巩固会话记忆"""
        await self.initialize()
        
        # 获取工作记忆
        working_memories = self.hierarchy_manager.working_memory.get_all()
        
        # 提升重要的工作记忆
        for memory in working_memories:
            await self.hierarchy_manager.promote_memory(memory)
            
        # 获取情景记忆
        episodic_memories = await self.hierarchy_manager.episodic_memory.retrieve(
            session_id=session_id,
            limit=100
        )
        
        # 提取语义知识
        for memory in episodic_memories:
            if self.hierarchy_manager._should_promote_to_semantic(memory):
                semantic = await self.hierarchy_manager.semantic_memory.extract_knowledge(
                    memory
                )
                if semantic:
                    await self.hierarchy_manager.semantic_memory.store(semantic)
                    
    async def cleanup_old_memories(
        self,
        days_old: int = 30,
        min_importance: float = 0.3
    ):
        """清理旧记忆"""
        await self.initialize()
        
        cutoff_date = utc_now() - timedelta(days=days_old)
        
        filters = MemoryFilters(
            created_before=cutoff_date,
            max_importance=min_importance
        )
        
        old_memories = await self.hierarchy_manager.storage.search_memories(
            filters,
            limit=1000
        )
        
        for memory in old_memories:
            if memory.access_count < 2:  # 很少访问的旧记忆
                await self.delete_memory(memory.id)
                
        logger.info(f"清理了 {len(old_memories)} 个旧记忆")

# 全局服务实例
memory_service = MemoryService()
from src.core.logging import get_logger
