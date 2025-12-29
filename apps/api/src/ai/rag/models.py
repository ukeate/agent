"""
RAG系统数据模型
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import uuid

@dataclass
class DocumentMetadata:
    """文档元数据"""
    source: str = ""
    title: str = ""
    author: str = ""
    language: str = "zh"
    document_type: str = "text"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """文档模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    chunks: List['DocumentChunk'] = field(default_factory=list)
    
    def add_chunk(self, chunk: 'DocumentChunk'):
        """添加文档块"""
        chunk.document_id = self.id
        self.chunks.append(chunk)
    
    def get_chunk_count(self) -> int:
        """获取文档块数量"""
        return len(self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "source": self.metadata.source,
                "title": self.metadata.title,
                "author": self.metadata.author,
                "language": self.metadata.language,
                "document_type": self.metadata.document_type,
                "tags": self.metadata.tags,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "custom_fields": self.metadata.custom_fields
            },
            "chunk_count": self.get_chunk_count()
        }

@dataclass
class DocumentChunk:
    """文档块模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    
    def set_embedding(self, embedding: List[float]):
        """设置向量嵌入"""
        self.embedding = embedding
    
    def has_embedding(self) -> bool:
        """检查是否有向量嵌入"""
        return self.embedding is not None and len(self.embedding) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "has_embedding": self.has_embedding(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class RetrievalResult:
    """检索结果模型"""
    chunk: DocumentChunk
    score: float
    retrieval_method: str = "vector"  # vector, keyword, hybrid
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "explanation": self.explanation
        }

@dataclass
class QueryContext:
    """查询上下文"""
    query: str
    query_type: str = "question"  # question, keyword, semantic
    filters: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 5
    min_score: float = 0.0
    language: str = "zh"
    session_id: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexStats:
    """索引统计信息"""
    total_documents: int = 0
    total_chunks: int = 0
    indexed_chunks: int = 0
    average_chunk_size: float = 0.0
    languages: List[str] = field(default_factory=list)
    document_types: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "indexed_chunks": self.indexed_chunks,
            "indexing_progress": self.indexed_chunks / max(self.total_chunks, 1),
            "average_chunk_size": self.average_chunk_size,
            "languages": self.languages,
            "document_types": self.document_types,
            "last_updated": self.last_updated.isoformat()
        }

@dataclass
class RAGResponse:
    """RAG生成响应"""
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    query: str
    generation_method: str = "openai"
    processing_time: float = 0.0
    tokens_used: int = 0
    context_used: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "confidence": self.confidence,
            "query": self.query,
            "generation_method": self.generation_method,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "context_used": self.context_used,
            "source_count": len(self.sources)
        }
