"""记忆层级管理器"""
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import logging
from collections import deque

from .models import Memory, MemoryType, MemoryStatus
from .storage import MemoryStorage
from .config import MemoryConfig
from ..openai_client import get_openai_client

logger = logging.getLogger(__name__)


class WorkingMemoryBuffer:
    """工作记忆缓冲区"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.memory_map: Dict[str, Memory] = {}
        
    def add(self, memory: Memory):
        """添加到工作记忆"""
        if len(self.buffer) >= self.capacity:
            # 移除最旧的记忆
            oldest = self.buffer.popleft()
            if oldest.id in self.memory_map:
                del self.memory_map[oldest.id]
                
        self.buffer.append(memory)
        self.memory_map[memory.id] = memory
        
    def get(self, memory_id: str) -> Optional[Memory]:
        """获取工作记忆"""
        return self.memory_map.get(memory_id)
        
    def get_all(self) -> List[Memory]:
        """获取所有工作记忆"""
        return list(self.buffer)
        
    def clear(self):
        """清空工作记忆"""
        self.buffer.clear()
        self.memory_map.clear()
        
    def is_full(self) -> bool:
        """检查是否已满"""
        return len(self.buffer) >= self.capacity


class EpisodicMemoryStore:
    """情景记忆存储"""
    
    def __init__(self, storage: MemoryStorage, limit: int = 10000):
        self.storage = storage
        self.limit = limit
        
    async def store(self, memory: Memory) -> Memory:
        """存储情景记忆"""
        memory.type = MemoryType.EPISODIC
        return await self.storage.store_memory(memory)
        
    async def retrieve(
        self,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """检索情景记忆"""
        from .models import MemoryFilters
        
        filters = MemoryFilters(
            memory_types=[MemoryType.EPISODIC],
            session_id=session_id,
            status=[MemoryStatus.ACTIVE]
        )
        return await self.storage.search_memories(filters, limit=limit)
        
    async def consolidate(self, memories: List[Memory]) -> Optional[Memory]:
        """巩固多个情景记忆为一个"""
        if len(memories) < 2:
            return None
            
        # 合并内容
        combined_content = "\n---\n".join([m.content for m in memories])
        
        # 计算综合重要性
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # 创建巩固后的记忆
        consolidated = Memory(
            type=MemoryType.EPISODIC,
            content=f"巩固的情景记忆 ({len(memories)}个事件):\n{combined_content}",
            importance=min(1.0, avg_importance * 1.2),  # 巩固后提升重要性
            metadata={
                "consolidated": True,
                "source_count": len(memories),
                "source_ids": [m.id for m in memories]
            },
            session_id=memories[0].session_id,
            user_id=memories[0].user_id
        )
        
        return await self.storage.store_memory(consolidated)


class SemanticMemoryStore:
    """语义记忆存储"""
    
    def __init__(self, storage: MemoryStorage, limit: int = 5000):
        self.storage = storage
        self.limit = limit
        self.openai_client = get_openai_client()
        
    async def store(self, memory: Memory) -> Memory:
        """存储语义记忆"""
        memory.type = MemoryType.SEMANTIC
        
        # 生成嵌入向量
        if not memory.embedding:
            memory.embedding = await self._generate_embedding(memory.content)
            
        return await self.storage.store_memory(memory)
        
    async def retrieve(
        self,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """检索语义记忆"""
        from .models import MemoryFilters
        
        filters = MemoryFilters(
            memory_types=[MemoryType.SEMANTIC],
            status=[MemoryStatus.ACTIVE]
        )
        return await self.storage.search_memories(filters, limit=limit)
        
    async def extract_knowledge(self, episodic_memory: Memory) -> Optional[Memory]:
        """从情景记忆中提取语义知识"""
        try:
            # 使用LLM提取关键知识点
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "从以下情景记忆中提取关键知识点和概念，以简洁的形式总结。"
                    },
                    {
                        "role": "user",
                        "content": episodic_memory.content
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            knowledge_content = response.choices[0].message.content
            
            # 创建语义记忆
            semantic_memory = Memory(
                type=MemoryType.SEMANTIC,
                content=knowledge_content,
                importance=min(1.0, episodic_memory.importance * 1.5),
                metadata={
                    "extracted_from": episodic_memory.id,
                    "extraction_method": "llm"
                },
                session_id=episodic_memory.session_id,
                user_id=episodic_memory.user_id,
                tags=episodic_memory.tags + ["extracted"]
            )
            
            # 生成嵌入向量
            semantic_memory.embedding = await self._generate_embedding(knowledge_content)
            
            return semantic_memory
            
        except Exception as e:
            logger.error(f"提取语义知识失败: {e}")
            return None
            
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []


class MemoryHierarchyManager:
    """记忆层级管理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.storage = MemoryStorage(self.config)
        
        # 初始化各层级存储
        self.working_memory = WorkingMemoryBuffer(self.config.working_memory_capacity)
        self.episodic_memory = EpisodicMemoryStore(
            self.storage, 
            self.config.episodic_memory_limit
        )
        self.semantic_memory = SemanticMemoryStore(
            self.storage,
            self.config.semantic_memory_limit
        )
        
    async def initialize(self):
        """初始化管理器"""
        await self.storage.initialize()
        
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.WORKING,
        **kwargs
    ) -> Memory:
        """添加新记忆"""
        memory = Memory(
            type=memory_type,
            content=content,
            **kwargs
        )
        
        if memory_type == MemoryType.WORKING:
            # 添加到工作记忆
            self.working_memory.add(memory)
            
            # 如果工作记忆满了，考虑提升
            if self.working_memory.is_full():
                await self._process_working_memory()
                
        elif memory_type == MemoryType.EPISODIC:
            memory = await self.episodic_memory.store(memory)
            
        elif memory_type == MemoryType.SEMANTIC:
            memory = await self.semantic_memory.store(memory)
            
        return memory
        
    async def promote_memory(self, memory: Memory):
        """提升记忆层级"""
        if memory.type == MemoryType.WORKING:
            if self._should_promote_to_episodic(memory):
                # 提升到情景记忆
                memory.type = MemoryType.EPISODIC
                await self.episodic_memory.store(memory)
                logger.info(f"记忆 {memory.id} 提升到情景记忆")
                
        elif memory.type == MemoryType.EPISODIC:
            if self._should_promote_to_semantic(memory):
                # 提取语义知识
                semantic = await self.semantic_memory.extract_knowledge(memory)
                if semantic:
                    await self.semantic_memory.store(semantic)
                    logger.info(f"从记忆 {memory.id} 提取语义知识")
                    
    def _should_promote_to_episodic(self, memory: Memory) -> bool:
        """判断是否应提升到情景记忆"""
        # 基于重要性和访问次数
        return (
            memory.importance > 0.6 or
            memory.access_count > 3
        )
        
    def _should_promote_to_semantic(self, memory: Memory) -> bool:
        """判断是否应提升到语义记忆"""
        # 基于重要性、访问次数和时间
        age = utc_now() - memory.created_at
        return (
            memory.importance > 0.7 and
            memory.access_count > self.config.consolidation_threshold and
            age > timedelta(hours=1)
        )
        
    async def _process_working_memory(self):
        """处理工作记忆溢出"""
        memories = self.working_memory.get_all()
        
        # 按重要性排序
        sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)
        
        # 保留重要的，提升到情景记忆
        for memory in sorted_memories[:10]:  # 保留前10个
            if self._should_promote_to_episodic(memory):
                await self.promote_memory(memory)
                
        # 清理最不重要的
        self.working_memory.clear()
        
        # 保留最重要的重新加入
        for memory in sorted_memories[:50]:  # 保留前50个
            self.working_memory.add(memory)
            
    async def get_relevant_memories(
        self,
        context: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """获取相关记忆"""
        relevant_memories = []
        
        if not memory_types:
            memory_types = list(MemoryType)
            
        # 从各层级获取记忆
        if MemoryType.WORKING in memory_types:
            working = self.working_memory.get_all()
            relevant_memories.extend(working[:limit])
            
        if MemoryType.EPISODIC in memory_types:
            episodic = await self.episodic_memory.retrieve(limit=limit)
            relevant_memories.extend(episodic)
            
        if MemoryType.SEMANTIC in memory_types:
            semantic = await self.semantic_memory.retrieve(limit=limit)
            relevant_memories.extend(semantic)
            
        # 按相关性排序(简单实现，后续会有更复杂的召回机制)
        relevant_memories.sort(key=lambda m: m.importance, reverse=True)
        
        return relevant_memories[:limit]
        
    async def cleanup(self):
        """清理资源"""
        await self.storage.cleanup()