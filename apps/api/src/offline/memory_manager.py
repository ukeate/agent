"""
离线记忆管理器

实现离线记忆存储，支持：
- 本地记忆查询
- 记忆压缩策略
- 记忆优先级管理
- 记忆同步机制
"""

import json
import sqlite3
import gzip
from src.core.utils import secure_pickle as pickle
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from contextlib import contextmanager
from src.models.schemas.offline import VectorClock
from src.core.config import get_settings

from src.core.logging import get_logger
class MemoryType(str, Enum):
    """记忆类型枚举"""
    CONVERSATION = "conversation"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"

class MemoryPriority(int, Enum):
    """记忆优先级枚举"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ARCHIVE = 5

@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    session_id: str
    memory_type: MemoryType
    content: str
    context: Dict[str, Any]
    embedding: Optional[List[float]] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    tags: List[str] = None
    
    # 时间信息
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    
    # 同步信息
    vector_clock: Optional[VectorClock] = None
    is_synced: bool = False
    last_synced: Optional[datetime] = None
    
    # 压缩信息
    is_compressed: bool = False
    original_size: int = 0
    compressed_size: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()
        if self.last_accessed is None:
            self.last_accessed = utc_now()
        if self.tags is None:
            self.tags = []

@dataclass
class MemoryQuery:
    """记忆查询"""
    query_text: str
    memory_types: List[MemoryType] = None
    tags: List[str] = None
    time_range: Tuple[datetime, datetime] = None
    priority_filter: List[MemoryPriority] = None
    limit: int = 10
    similarity_threshold: float = 0.7

@dataclass
class MemorySearchResult:
    """记忆搜索结果"""
    entry: MemoryEntry
    similarity_score: float
    rank: int

class MemoryCompressionStrategy:
    """记忆压缩策略"""
    
    @staticmethod
    def compress_content(content: str, method: str = "gzip") -> Tuple[bytes, float]:
        """压缩记忆内容"""
        content_bytes = content.encode('utf-8')
        original_size = len(content_bytes)
        
        if method == "gzip":
            compressed = gzip.compress(content_bytes, compresslevel=9)
        else:
            compressed = content_bytes
        
        compression_ratio = len(compressed) / original_size
        return compressed, compression_ratio
    
    @staticmethod
    def decompress_content(compressed_data: bytes, method: str = "gzip") -> str:
        """解压记忆内容"""
        if method == "gzip":
            decompressed = gzip.decompress(compressed_data)
        else:
            decompressed = compressed_data
        
        return decompressed.decode('utf-8')
    
    @staticmethod
    def should_compress(content: str, min_size: int = 1024) -> bool:
        """判断是否应该压缩"""
        return len(content.encode('utf-8')) > min_size

class OfflineMemoryManager:
    """离线记忆管理器"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.storage_path = Path(storage_path or Path.cwd() / ".offline_storage" / "memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "memory.db"
        self.compression_strategy = MemoryCompressionStrategy()
        
        # 内存缓存
        self._cache: Dict[str, MemoryEntry] = {}
        self._cache_size_limit = 1000
        
        # 嵌入向量缓存
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT,
                    compressed_content BLOB,
                    context TEXT NOT NULL,
                    embedding BLOB,
                    priority INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    vector_clock TEXT,
                    is_synced BOOLEAN DEFAULT FALSE,
                    last_synced TEXT,
                    is_compressed BOOLEAN DEFAULT FALSE,
                    original_size INTEGER DEFAULT 0,
                    compressed_size INTEGER DEFAULT 0
                )
            """)
            
            # 创建索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)",
                "CREATE INDEX IF NOT EXISTS idx_memory_priority ON memory_entries(priority)",
                "CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memory_entries(last_accessed)",
                "CREATE INDEX IF NOT EXISTS idx_memory_synced ON memory_entries(is_synced)",
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _serialize_embedding(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """序列化嵌入向量"""
        if embedding is None:
            return None
        return pickle.dumps(np.array(embedding))
    
    def _deserialize_embedding(self, embedding_blob: Optional[bytes]) -> Optional[List[float]]:
        """反序列化嵌入向量"""
        if embedding_blob is None:
            return None
        return pickle.loads(embedding_blob).tolist()
    
    def store_memory(self, memory: MemoryEntry) -> bool:
        """存储记忆"""
        try:
            # 检查是否需要压缩
            content_to_store = memory.content
            compressed_content = None
            
            if self.compression_strategy.should_compress(memory.content):
                compressed_data, compression_ratio = self.compression_strategy.compress_content(
                    memory.content
                )
                compressed_content = compressed_data
                memory.is_compressed = True
                memory.original_size = len(memory.content.encode('utf-8'))
                memory.compressed_size = len(compressed_data)
                content_to_store = None  # 不存储原始内容
            
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_entries (
                        id, session_id, memory_type, content, compressed_content,
                        context, embedding, priority, tags, created_at, last_accessed,
                        access_count, vector_clock, is_synced, last_synced,
                        is_compressed, original_size, compressed_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.session_id, memory.memory_type.value,
                    content_to_store, compressed_content,
                    json.dumps(memory.context),
                    self._serialize_embedding(memory.embedding),
                    memory.priority.value, json.dumps(memory.tags),
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    json.dumps(memory.vector_clock.model_dump()) if memory.vector_clock else None,
                    memory.is_synced,
                    memory.last_synced.isoformat() if memory.last_synced else None,
                    memory.is_compressed, memory.original_size, memory.compressed_size
                ))
            
            # 添加到缓存
            self._add_to_cache(memory)
            
            return True
            
        except Exception as e:
            self.logger.error("保存记忆失败", memory_id=memory.id, error=str(e))
            return False
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """检索记忆"""
        # 首先检查缓存
        if memory_id in self._cache:
            memory = self._cache[memory_id]
            memory.last_accessed = utc_now()
            memory.access_count += 1
            self._update_access_stats(memory)
            return memory
        
        # 从数据库加载
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memory_entries WHERE id = ?",
                (memory_id,)
            ).fetchone()
            
            if not row:
                return None
            
            memory = self._row_to_memory(row)
            
            # 更新访问统计
            memory.last_accessed = utc_now()
            memory.access_count += 1
            self._update_access_stats(memory)
            
            # 添加到缓存
            self._add_to_cache(memory)
            
            return memory
    
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """将数据库行转换为记忆条目"""
        # 处理内容（可能被压缩）
        if row['is_compressed'] and row['compressed_content']:
            content = self.compression_strategy.decompress_content(row['compressed_content'])
        else:
            content = row['content']
        
        return MemoryEntry(
            id=row['id'],
            session_id=row['session_id'],
            memory_type=MemoryType(row['memory_type']),
            content=content,
            context=json.loads(row['context']),
            embedding=self._deserialize_embedding(row['embedding']),
            priority=MemoryPriority(row['priority']),
            tags=json.loads(row['tags']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            access_count=row['access_count'],
            vector_clock=VectorClock(**json.loads(row['vector_clock'])) if row['vector_clock'] else None,
            is_synced=row['is_synced'],
            last_synced=datetime.fromisoformat(row['last_synced']) if row['last_synced'] else None,
            is_compressed=row['is_compressed'],
            original_size=row['original_size'],
            compressed_size=row['compressed_size']
        )
    
    def search_memories(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """搜索记忆"""
        conditions = ["1=1"]
        params = []
        
        # 构建查询条件
        if query.memory_types:
            placeholders = ','.join('?' * len(query.memory_types))
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend([mt.value for mt in query.memory_types])
        
        if query.priority_filter:
            placeholders = ','.join('?' * len(query.priority_filter))
            conditions.append(f"priority IN ({placeholders})")
            params.extend([p.value for p in query.priority_filter])
        
        if query.time_range:
            conditions.append("created_at BETWEEN ? AND ?")
            params.extend([
                query.time_range[0].isoformat(),
                query.time_range[1].isoformat()
            ])
        
        # 标签过滤
        if query.tags:
            tag_conditions = []
            for tag in query.tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")
        
        sql = f"""
            SELECT * FROM memory_entries 
            WHERE {' AND '.join(conditions)}
            ORDER BY last_accessed DESC, priority ASC
            LIMIT ?
        """
        params.append(query.limit * 2)  # 获取更多结果用于后续过滤
        
        results = []
        with self.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            
            for row in rows:
                memory = self._row_to_memory(row)
                
                # 计算相似度分数
                similarity_score = self._calculate_similarity(query.query_text, memory)
                
                if similarity_score >= query.similarity_threshold:
                    results.append(MemorySearchResult(
                        entry=memory,
                        similarity_score=similarity_score,
                        rank=len(results) + 1
                    ))
        
        # 按相似度排序并限制结果数量
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:query.limit]
    
    def _calculate_similarity(self, query_text: str, memory: MemoryEntry) -> float:
        """计算查询与记忆的相似度"""
        # 简单的文本相似度计算（在实际应用中应该使用更复杂的方法）
        query_words = set(query_text.lower().split())
        memory_words = set(memory.content.lower().split())
        
        if not query_words or not memory_words:
            return 0.0
        
        intersection = query_words.intersection(memory_words)
        union = query_words.union(memory_words)
        
        jaccard_similarity = len(intersection) / len(union)
        
        # 根据记忆类型和优先级调整分数
        type_bonus = {
            MemoryType.WORKING: 0.2,
            MemoryType.CONVERSATION: 0.1,
            MemoryType.FACTUAL: 0.15,
            MemoryType.PROCEDURAL: 0.1,
            MemoryType.EPISODIC: 0.05,
            MemoryType.SEMANTIC: 0.1
        }.get(memory.memory_type, 0.0)
        
        priority_bonus = {
            MemoryPriority.CRITICAL: 0.3,
            MemoryPriority.HIGH: 0.2,
            MemoryPriority.MEDIUM: 0.1,
            MemoryPriority.LOW: 0.05,
            MemoryPriority.ARCHIVE: 0.0
        }.get(memory.priority, 0.0)
        
        # 访问频率奖励
        access_bonus = min(memory.access_count * 0.01, 0.1)
        
        final_score = min(jaccard_similarity + type_bonus + priority_bonus + access_bonus, 1.0)
        return final_score
    
    def get_memories_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEntry]:
        """获取会话的所有记忆"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM memory_entries 
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    def get_memories_by_type(self, memory_type: MemoryType, limit: int = 100) -> List[MemoryEntry]:
        """获取特定类型的记忆"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM memory_entries 
                WHERE memory_type = ?
                ORDER BY last_accessed DESC
                LIMIT ?
            """, (memory_type.value, limit)).fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    def update_memory_priority(self, memory_id: str, new_priority: MemoryPriority) -> bool:
        """更新记忆优先级"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE memory_entries 
                    SET priority = ?, last_accessed = ?
                    WHERE id = ?
                """, (new_priority.value, utc_now().isoformat(), memory_id))
                
                success = cursor.rowcount > 0
                
                # 更新缓存
                if success and memory_id in self._cache:
                    self._cache[memory_id].priority = new_priority
                    self._cache[memory_id].last_accessed = utc_now()
                
                return success
                
        except Exception as e:
            self.logger.error("更新记忆优先级失败", memory_id=memory_id, error=str(e))
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM memory_entries WHERE id = ?",
                    (memory_id,)
                )
                
                success = cursor.rowcount > 0
                
                # 从缓存移除
                if success and memory_id in self._cache:
                    del self._cache[memory_id]
                
                return success
                
        except Exception as e:
            self.logger.error("删除记忆失败", memory_id=memory_id, error=str(e))
            return False
    
    def compress_old_memories(self, days_threshold: int = 30) -> int:
        """压缩旧记忆"""
        cutoff_date = (utc_now() - timedelta(days=days_threshold)).isoformat()
        compressed_count = 0
        
        with self.get_connection() as conn:
            # 查找需要压缩的记忆
            rows = conn.execute("""
                SELECT id, content FROM memory_entries 
                WHERE last_accessed < ? AND is_compressed = FALSE AND content IS NOT NULL
            """, (cutoff_date,)).fetchall()
            
            for row in rows:
                memory_id, content = row['id'], row['content']
                
                if self.compression_strategy.should_compress(content):
                    try:
                        compressed_data, compression_ratio = self.compression_strategy.compress_content(content)
                        
                        conn.execute("""
                            UPDATE memory_entries 
                            SET content = NULL, compressed_content = ?, is_compressed = TRUE,
                                original_size = ?, compressed_size = ?
                            WHERE id = ?
                        """, (
                            compressed_data, len(content.encode('utf-8')),
                            len(compressed_data), memory_id
                        ))
                        
                        compressed_count += 1
                        
                        # 从缓存移除以强制重新加载
                        if memory_id in self._cache:
                            del self._cache[memory_id]
                        
                    except Exception as e:
                        self.logger.error("压缩记忆失败", memory_id=memory_id, error=str(e))
        
        return compressed_count
    
    def _add_to_cache(self, memory: MemoryEntry):
        """添加记忆到缓存"""
        # 如果缓存满了，移除最不常用的记忆
        if len(self._cache) >= self._cache_size_limit:
            # 找到最不常用的记忆
            lru_memory = min(
                self._cache.values(),
                key=lambda m: (m.last_accessed, m.access_count)
            )
            del self._cache[lru_memory.id]
        
        self._cache[memory.id] = memory
    
    def _update_access_stats(self, memory: MemoryEntry):
        """更新访问统计"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE memory_entries 
                    SET last_accessed = ?, access_count = ?
                    WHERE id = ?
                """, (
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    memory.id
                ))
        except Exception as e:
            self.logger.error("更新记忆访问统计失败", memory_id=memory.id, error=str(e))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        with self.get_connection() as conn:
            # 总体统计
            total_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    SUM(original_size) as total_original_size,
                    SUM(compressed_size) as total_compressed_size,
                    AVG(access_count) as avg_access_count,
                    COUNT(CASE WHEN is_compressed THEN 1 END) as compressed_memories
                FROM memory_entries
            """).fetchone()
            
            # 按类型统计
            type_stats = conn.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memory_entries
                GROUP BY memory_type
            """).fetchall()
            
            # 按优先级统计
            priority_stats = conn.execute("""
                SELECT priority, COUNT(*) as count
                FROM memory_entries
                GROUP BY priority
            """).fetchall()
            
            return {
                'total_memories': total_stats['total_memories'],
                'total_original_size_bytes': total_stats['total_original_size'] or 0,
                'total_compressed_size_bytes': total_stats['total_compressed_size'] or 0,
                'compression_ratio': (
                    total_stats['total_compressed_size'] / total_stats['total_original_size']
                    if total_stats['total_original_size'] else 1.0
                ),
                'average_access_count': total_stats['avg_access_count'] or 0,
                'compressed_memories': total_stats['compressed_memories'],
                'cached_memories': len(self._cache),
                'memory_types': {row['memory_type']: row['count'] for row in type_stats},
                'priorities': {row['priority']: row['count'] for row in priority_stats}
            }
    
    def cleanup_old_memories(self, days_threshold: int = 90, keep_important: bool = True) -> int:
        """清理旧记忆"""
        cutoff_date = (utc_now() - timedelta(days=days_threshold)).isoformat()
        
        conditions = ["last_accessed < ?"]
        params = [cutoff_date]
        
        if keep_important:
            # 保留重要记忆
            conditions.append("priority > ?")
            params.append(MemoryPriority.HIGH.value)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(f"""
                    DELETE FROM memory_entries 
                    WHERE {' AND '.join(conditions)}
                """, params)
                
                deleted_count = cursor.rowcount
                
                # 清理缓存中的相关条目
                to_remove = [
                    memory_id for memory_id, memory in self._cache.items()
                    if memory.last_accessed < datetime.fromisoformat(cutoff_date)
                ]
                
                for memory_id in to_remove:
                    del self._cache[memory_id]
                
                return deleted_count
                
        except Exception as e:
            self.logger.error("清理旧记忆失败", error=str(e))
            return 0
