"""基于上下文的记忆召回系统"""
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from .models import Memory, MemoryType, MemoryStatus
from .storage import MemoryStorage
from .config import MemoryConfig
from src.ai.openai_client import get_openai_client
from offline.memory_manager import (
    OfflineMemoryManager, MemoryEntry, MemoryQuery, 
    MemoryType as OfflineMemoryType, MemoryPriority
)
from models.schemas.offline import OfflineMode, NetworkStatus

logger = logging.getLogger(__name__)


class ContextAwareRecall:
    """基于上下文的记忆召回器"""
    
    def __init__(self, storage: MemoryStorage, config: Optional[MemoryConfig] = None):
        self.storage = storage
        self.config = config or MemoryConfig()
        self.openai_client = get_openai_client()
        
        # 离线记忆管理器
        self.offline_memory_manager = OfflineMemoryManager()
        self.offline_mode = OfflineMode.ONLINE
        self.network_status = NetworkStatus.CONNECTED
        
    def set_offline_mode(self, mode: OfflineMode, network_status: NetworkStatus = NetworkStatus.UNKNOWN):
        """设置离线模式"""
        self.offline_mode = mode
        self.network_status = network_status
    
    async def recall_relevant_memories(
        self,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """召回相关记忆，返回记忆和相关性评分"""
        
        # 根据离线模式选择召回策略
        if self.offline_mode == OfflineMode.OFFLINE or self.network_status == NetworkStatus.DISCONNECTED:
            return await self._offline_recall(context, session_id, user_id, memory_types, limit)
        elif self.offline_mode == OfflineMode.ONLINE and self.network_status == NetworkStatus.CONNECTED:
            return await self._online_recall(context, session_id, user_id, memory_types, limit)
        else:
            # 混合模式：优先在线，降级到离线
            try:
                return await self._online_recall(context, session_id, user_id, memory_types, limit)
            except Exception as e:
                logger.warning(f"在线召回失败，降级到离线模式: {e}")
                return await self._offline_recall(context, session_id, user_id, memory_types, limit)
    
    async def _online_recall(
        self,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """在线记忆召回（原有逻辑）"""
        # 生成上下文嵌入
        context_embedding = await self._generate_embedding(context)
        
        # 多维度检索
        vector_results = await self._vector_search(
            context_embedding, 
            session_id, 
            user_id,
            memory_types,
            limit * 2  # 获取更多候选
        )
        
        temporal_results = await self._temporal_search(
            context,
            session_id,
            user_id,
            memory_types,
            limit
        )
        
        entity_results = await self._entity_search(
            context,
            session_id,
            user_id,
            memory_types,
            limit
        )
        
        # 融合和排序结果
        merged_results = self._rank_and_merge(
            vector_results,
            temporal_results,
            entity_results,
            context_embedding
        )
        
        return merged_results[:limit]
    
    async def _offline_recall(
        self,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """离线记忆召回"""
        # 转换记忆类型
        offline_types = []
        if memory_types:
            type_mapping = {
                MemoryType.CONVERSATION: OfflineMemoryType.CONVERSATION,
                MemoryType.FACTUAL: OfflineMemoryType.FACTUAL,
                MemoryType.PROCEDURAL: OfflineMemoryType.PROCEDURAL,
                MemoryType.EPISODIC: OfflineMemoryType.EPISODIC,
                MemoryType.SEMANTIC: OfflineMemoryType.SEMANTIC
            }
            offline_types = [type_mapping.get(mt, OfflineMemoryType.FACTUAL) for mt in memory_types]
        
        # 构建离线查询
        query = MemoryQuery(
            query_text=context,
            memory_types=offline_types if offline_types else None,
            limit=limit,
            similarity_threshold=0.3  # 降低阈值以获取更多结果
        )
        
        # 执行离线搜索
        search_results = self.offline_memory_manager.search_memories(query)
        
        # 转换结果格式
        results = []
        for search_result in search_results:
            offline_memory = search_result.entry
            
            # 转换为在线记忆格式
            online_memory = self._convert_offline_to_online_memory(offline_memory)
            results.append((online_memory, search_result.similarity_score))
        
        return results
    
    def _convert_offline_to_online_memory(self, offline_memory: MemoryEntry) -> Memory:
        """将离线记忆转换为在线记忆格式"""
        # 转换记忆类型
        type_mapping = {
            OfflineMemoryType.CONVERSATION: MemoryType.CONVERSATION,
            OfflineMemoryType.FACTUAL: MemoryType.FACTUAL,
            OfflineMemoryType.PROCEDURAL: MemoryType.PROCEDURAL,
            OfflineMemoryType.EPISODIC: MemoryType.EPISODIC,
            OfflineMemoryType.SEMANTIC: MemoryType.SEMANTIC,
            OfflineMemoryType.WORKING: MemoryType.CONVERSATION  # 默认映射
        }
        
        online_type = type_mapping.get(offline_memory.memory_type, MemoryType.FACTUAL)
        
        # 创建在线记忆对象（这里需要根据实际的Memory类结构调整）
        from .models import Memory
        
        online_memory = Memory(
            id=offline_memory.id,
            session_id=offline_memory.session_id,
            user_id=offline_memory.context.get('user_id', ''),
            content=offline_memory.content,
            type=online_type,
            embedding=offline_memory.embedding,
            tags=offline_memory.tags,
            created_at=offline_memory.created_at,
            last_accessed=offline_memory.last_accessed,
            importance=self._priority_to_importance(offline_memory.priority),
            status=MemoryStatus.ACTIVE
        )
        
        return online_memory
    
    def _priority_to_importance(self, priority: MemoryPriority) -> float:
        """将离线优先级转换为在线重要性"""
        priority_mapping = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.6,
            MemoryPriority.LOW: 0.4,
            MemoryPriority.ARCHIVE: 0.2
        }
        return priority_mapping.get(priority, 0.6)
        
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
            return [0.0] * self.config.vector_dimension
            
    async def _vector_search(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20
    ) -> List[Tuple[Memory, float]]:
        """向量相似度搜索"""
        if not self.storage.vector_store:
            return []
            
        try:
            # 构建过滤条件
            filter_conditions = []
            
            if session_id:
                filter_conditions.append(
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                )
                
            if user_id:
                filter_conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                )
                
            if memory_types:
                type_values = [t.value for t in memory_types]
                filter_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(any=type_values)
                    )
                )
                
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # 执行向量搜索
            search_result = await asyncio.to_thread(
                self.storage.vector_store.search,
                collection_name=self.config.qdrant_collection,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter
            )
            
            # 获取记忆对象
            results = []
            for point in search_result:
                memory = await self.storage.get_memory(point.id)
                if memory:
                    results.append((memory, point.score))
                    
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
            
    async def _temporal_search(
        self,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """时间相关性搜索"""
        from .models import MemoryFilters
        
        # 获取最近的记忆
        filters = MemoryFilters(
            session_id=session_id,
            user_id=user_id,
            memory_types=memory_types,
            created_after=datetime.utcnow() - timedelta(hours=24),  # 最近24小时
            status=[MemoryStatus.ACTIVE]
        )
        
        recent_memories = await self.storage.search_memories(filters, limit=limit)
        
        # 计算时间衰减评分
        results = []
        current_time = datetime.utcnow()
        
        for memory in recent_memories:
            time_diff = (current_time - memory.last_accessed).total_seconds()
            # 指数衰减
            temporal_score = np.exp(-time_diff / self.config.decay_constant)
            results.append((memory, temporal_score))
            
        return results
        
    async def _entity_search(
        self,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """实体相关性搜索"""
        # 提取上下文中的关键实体
        entities = await self._extract_entities(context)
        
        if not entities:
            return []
            
        from .models import MemoryFilters
        
        # 搜索包含这些实体的记忆
        filters = MemoryFilters(
            session_id=session_id,
            user_id=user_id,
            memory_types=memory_types,
            tags=entities,  # 使用标签匹配实体
            status=[MemoryStatus.ACTIVE]
        )
        
        entity_memories = await self.storage.search_memories(filters, limit=limit)
        
        # 计算实体匹配评分
        results = []
        for memory in entity_memories:
            # 计算匹配的实体数量
            matched_entities = set(memory.tags) & set(entities)
            entity_score = len(matched_entities) / max(len(entities), 1)
            results.append((memory, entity_score))
            
        return results
        
    async def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取关键实体"""
        try:
            # 使用LLM提取实体
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "从文本中提取关键实体(人名、地点、概念等)，以JSON数组格式返回。"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=100,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("entities", [])
            
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            # 简单的关键词提取作为后备
            import re
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b', text)
            return list(set(words))[:5]
            
    def _rank_and_merge(
        self,
        vector_results: List[Tuple[Memory, float]],
        temporal_results: List[Tuple[Memory, float]],
        entity_results: List[Tuple[Memory, float]],
        context_embedding: List[float]
    ) -> List[Tuple[Memory, float]]:
        """融合多维度搜索结果并排序"""
        # 合并所有结果
        memory_scores: Dict[str, Dict[str, float]] = {}
        
        # 处理向量搜索结果
        for memory, score in vector_results:
            if memory.id not in memory_scores:
                memory_scores[memory.id] = {
                    "memory": memory,
                    "vector_score": 0.0,
                    "temporal_score": 0.0,
                    "entity_score": 0.0
                }
            memory_scores[memory.id]["vector_score"] = score
            
        # 处理时间搜索结果
        for memory, score in temporal_results:
            if memory.id not in memory_scores:
                memory_scores[memory.id] = {
                    "memory": memory,
                    "vector_score": 0.0,
                    "temporal_score": 0.0,
                    "entity_score": 0.0
                }
            memory_scores[memory.id]["temporal_score"] = score
            
        # 处理实体搜索结果
        for memory, score in entity_results:
            if memory.id not in memory_scores:
                memory_scores[memory.id] = {
                    "memory": memory,
                    "vector_score": 0.0,
                    "temporal_score": 0.0,
                    "entity_score": 0.0
                }
            memory_scores[memory.id]["entity_score"] = score
            
        # 计算综合评分
        results = []
        for memory_id, scores in memory_scores.items():
            memory = scores["memory"]
            
            # 加权组合不同维度的评分
            final_score = (
                scores["vector_score"] * 0.5 +  # 向量相似度权重最高
                scores["temporal_score"] * 0.2 +  # 时间相关性
                scores["entity_score"] * 0.2 +  # 实体匹配
                memory.importance * 0.1  # 记忆重要性
            )
            
            results.append((memory, final_score))
            
        # 按综合评分排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
        
    async def recall_by_similarity(
        self,
        reference_memory: Memory,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """基于参考记忆召回相似记忆"""
        if not reference_memory.embedding:
            # 如果没有嵌入，使用内容生成
            reference_memory.embedding = await self._generate_embedding(
                reference_memory.content
            )
            
        # 使用向量搜索找相似记忆
        similar_memories = await self._vector_search(
            reference_memory.embedding,
            session_id=reference_memory.session_id,
            user_id=reference_memory.user_id,
            limit=limit + 1  # 多获取一个以排除自己
        )
        
        # 排除参考记忆自己
        results = [
            (mem, score) 
            for mem, score in similar_memories 
            if mem.id != reference_memory.id
        ]
        
        return results[:limit]
        
    async def recall_chain(
        self,
        start_memory: Memory,
        max_depth: int = 3,
        max_per_level: int = 3
    ) -> List[Memory]:
        """链式召回相关记忆"""
        visited = set()
        result_chain = []
        current_level = [start_memory]
        
        for depth in range(max_depth):
            next_level = []
            
            for memory in current_level:
                if memory.id in visited:
                    continue
                    
                visited.add(memory.id)
                result_chain.append(memory)
                
                # 获取相关记忆
                related = await self.recall_by_similarity(
                    memory,
                    limit=max_per_level
                )
                
                for related_memory, _ in related:
                    if related_memory.id not in visited:
                        next_level.append(related_memory)
                        
            current_level = next_level[:max_per_level * len(current_level)]
            
            if not current_level:
                break
                
        return result_chain