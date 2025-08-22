"""记忆持久化存储层"""
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
import redis.asyncio as redis
import asyncpg
from asyncpg.pool import Pool

from .models import Memory, MemoryType, MemoryStatus, MemoryFilters
from .config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryStorage:
    """记忆持久化存储管理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.vector_store: Optional[QdrantClient] = None
        self.redis_cache: Optional[redis.Redis] = None
        self.postgres_pool: Optional[Pool] = None
        self._initialized = False
        
    async def initialize(self):
        """初始化存储连接"""
        if self._initialized:
            return
            
        try:
            # 初始化Qdrant向量存储
            self.vector_store = QdrantClient(
                url=self.config.qdrant_url,
                check_compatibility=False  # 跳过版本兼容性检查
            )
            
            # 创建集合(如果不存在)
            collections = await asyncio.to_thread(
                self.vector_store.get_collections
            )
            collection_names = [c.name for c in collections.collections]
            
            if self.config.qdrant_collection not in collection_names:
                await asyncio.to_thread(
                    self.vector_store.create_collection,
                    collection_name=self.config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.config.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建Qdrant集合: {self.config.qdrant_collection}")
            
            # 初始化Redis缓存
            self.redis_cache = await redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # 初始化PostgreSQL连接池
            self.postgres_pool = await asyncpg.create_pool(
                self.config.db_url,
                min_size=2,
                max_size=10
            )
            
            # 创建记忆表(如果不存在)
            await self._create_tables()
            
            self._initialized = True
            logger.info("记忆存储层初始化完成")
            
        except Exception as e:
            logger.error(f"记忆存储层初始化失败: {e}")
            raise
            
    async def _create_tables(self):
        """创建数据库表"""
        async with self.postgres_pool.acquire() as conn:
            # 创建memories表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id VARCHAR(36) PRIMARY KEY,
                    type VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    importance FLOAT DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    decay_factor FLOAT DEFAULT 0.5,
                    status VARCHAR(20) DEFAULT 'active',
                    session_id VARCHAR(36),
                    user_id VARCHAR(36),
                    related_memories TEXT[] DEFAULT '{}',
                    tags TEXT[] DEFAULT '{}',
                    source VARCHAR(255)
                )
            ''')
            
            # 创建索引
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (session_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (type)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (created_at DESC)')
            
    async def store_memory(self, memory: Memory) -> Memory:
        """存储记忆"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # 1. 存储到PostgreSQL
            async with self.postgres_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO memories (
                        id, type, content, metadata, importance, access_count,
                        created_at, last_accessed, decay_factor, status,
                        session_id, user_id, related_memories, tags, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        importance = EXCLUDED.importance,
                        access_count = EXCLUDED.access_count,
                        last_accessed = EXCLUDED.last_accessed,
                        decay_factor = EXCLUDED.decay_factor,
                        status = EXCLUDED.status,
                        related_memories = EXCLUDED.related_memories,
                        tags = EXCLUDED.tags
                ''',
                memory.id, memory.type.value, memory.content,
                json.dumps(memory.metadata), memory.importance,
                memory.access_count, memory.created_at, memory.last_accessed,
                memory.decay_factor, memory.status.value, memory.session_id,
                memory.user_id, memory.related_memories, memory.tags,
                memory.source
                )
            
            # 2. 如果有嵌入向量，存储到Qdrant
            if memory.embedding:
                point = PointStruct(
                    id=memory.id,
                    vector=memory.embedding,
                    payload={
                        "type": memory.type.value,
                        "importance": memory.importance,
                        "session_id": memory.session_id,
                        "user_id": memory.user_id,
                        "tags": memory.tags,
                        "created_at": memory.created_at.isoformat()
                    }
                )
                await asyncio.to_thread(
                    self.vector_store.upsert,
                    collection_name=self.config.qdrant_collection,
                    points=[point]
                )
            
            # 3. 缓存到Redis(工作记忆)
            if memory.type == MemoryType.WORKING:
                cache_key = f"{self.config.cache_prefix}{memory.id}"
                await self.redis_cache.setex(
                    cache_key,
                    self.config.cache_ttl,
                    memory.json()
                )
            
            logger.info(f"记忆存储成功: {memory.id}")
            return memory
            
        except Exception as e:
            logger.error(f"记忆存储失败: {e}")
            raise
            
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # 1. 先从缓存查找
            cache_key = f"{self.config.cache_prefix}{memory_id}"
            cached = await self.redis_cache.get(cache_key)
            if cached:
                memory = Memory.parse_raw(cached)
                # 更新访问信息
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                await self.update_memory_access(memory_id)
                return memory
            
            # 2. 从数据库查找
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT * FROM memories WHERE id = $1',
                    memory_id
                )
                
            if not row:
                return None
                
            memory = self._row_to_memory(row)
            
            # 3. 更新访问信息
            await self.update_memory_access(memory_id)
            
            # 4. 缓存(如果是活跃记忆)
            if memory.status == MemoryStatus.ACTIVE:
                await self.redis_cache.setex(
                    cache_key,
                    self.config.cache_ttl,
                    memory.json()
                )
            
            return memory
            
        except Exception as e:
            logger.error(f"获取记忆失败 {memory_id}: {e}")
            return None
            
    async def update_memory_access(self, memory_id: str):
        """更新记忆访问信息"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute('''
                UPDATE memories 
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = $1
            ''', memory_id)
            
    async def search_memories(
        self,
        filters: Optional[MemoryFilters] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Memory]:
        """搜索记忆"""
        if not self._initialized:
            await self.initialize()
            
        query = 'SELECT * FROM memories WHERE 1=1'
        params = []
        param_count = 0
        
        if filters:
            if filters.memory_types:
                param_count += 1
                types = [t.value for t in filters.memory_types]
                query += f' AND type = ANY(${param_count})'
                params.append(types)
                
            if filters.status:
                param_count += 1
                statuses = [s.value for s in filters.status]
                query += f' AND status = ANY(${param_count})'
                params.append(statuses)
                
            if filters.min_importance is not None:
                param_count += 1
                query += f' AND importance >= ${param_count}'
                params.append(filters.min_importance)
                
            if filters.max_importance is not None:
                param_count += 1
                query += f' AND importance <= ${param_count}'
                params.append(filters.max_importance)
                
            if filters.session_id:
                param_count += 1
                query += f' AND session_id = ${param_count}'
                params.append(filters.session_id)
                
            if filters.user_id:
                param_count += 1
                query += f' AND user_id = ${param_count}'
                params.append(filters.user_id)
                
            if filters.tags:
                param_count += 1
                query += f' AND tags && ${param_count}'
                params.append(filters.tags)
                
        query += ' ORDER BY created_at DESC'
        param_count += 1
        query += f' LIMIT ${param_count}'
        params.append(limit)
        param_count += 1
        query += f' OFFSET ${param_count}'
        params.append(offset)
        
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return [self._row_to_memory(row) for row in rows]
        
    async def get_session_memories(
        self,
        session_id: str,
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """获取会话的所有记忆"""
        filters = MemoryFilters(
            session_id=session_id,
            memory_types=[memory_type] if memory_type else None
        )
        return await self.search_memories(filters, limit=1000)
        
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # 1. 标记为删除状态(软删除)
            async with self.postgres_pool.acquire() as conn:
                result = await conn.execute('''
                    UPDATE memories 
                    SET status = $1
                    WHERE id = $2
                ''', MemoryStatus.DELETED.value, memory_id)
                
            # 2. 从缓存删除
            cache_key = f"{self.config.cache_prefix}{memory_id}"
            await self.redis_cache.delete(cache_key)
            
            # 3. 从向量存储删除
            await asyncio.to_thread(
                self.vector_store.delete,
                collection_name=self.config.qdrant_collection,
                points_selector=[memory_id]
            )
            
            return result != "UPDATE 0"
            
        except Exception as e:
            logger.error(f"删除记忆失败 {memory_id}: {e}")
            return False
            
    async def bulk_insert(self, memories: List[Memory]) -> Dict[str, Any]:
        """批量插入记忆"""
        if not self._initialized:
            await self.initialize()
            
        success_count = 0
        failed_count = 0
        errors = []
        
        for memory in memories:
            try:
                await self.store_memory(memory)
                success_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(f"Memory {memory.id}: {str(e)}")
                
        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors
        }
        
    async def cleanup(self):
        """清理资源"""
        if self.redis_cache:
            await self.redis_cache.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
            
    def _row_to_memory(self, row) -> Memory:
        """数据库行转换为Memory对象"""
        return Memory(
            id=row['id'],
            type=MemoryType(row['type']),
            content=row['content'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            importance=row['importance'],
            access_count=row['access_count'],
            created_at=row['created_at'],
            last_accessed=row['last_accessed'],
            decay_factor=row['decay_factor'],
            status=MemoryStatus(row['status']),
            session_id=row['session_id'],
            user_id=row['user_id'],
            related_memories=row['related_memories'] or [],
            tags=row['tags'] or [],
            source=row['source']
        )
    
    async def close(self):
        """关闭存储连接，释放资源"""
        try:
            if self.redis_cache:
                await self.redis_cache.close()
                self.redis_cache = None
                
            if self.postgres_pool:
                await self.postgres_pool.close()
                self.postgres_pool = None
                
            # Qdrant客户端会自动管理连接
            self.vector_store = None
            
            self._initialized = False
            logger.info("内存存储连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭存储连接时出错: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()