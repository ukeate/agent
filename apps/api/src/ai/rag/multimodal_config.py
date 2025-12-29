"""多模态RAG系统配置"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class EmbeddingModel(str, Enum):
    """嵌入模型类型"""
    SENTENCE_TRANSFORMERS_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

class VectorStoreType(str, Enum):
    """向量存储类型"""
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"

class QueryType(str, Enum):
    """查询类型"""
    TEXT = "text"
    VISUAL = "visual"
    MIXED = "mixed"
    DOCUMENT = "document"

class MultimodalConfig(BaseModel):
    """多模态RAG配置"""

    model_config = ConfigDict(use_enum_values=True)
    
    # 向量存储配置
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.CHROMA,
        description="向量存储类型"
    )
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Chroma持久化目录"
    )
    qdrant_url: Optional[str] = Field(
        default="http://localhost:6333",
        description="Qdrant服务地址"
    )
    
    # 嵌入模型配置
    text_embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.SENTENCE_TRANSFORMERS_MINILM,
        description="文本嵌入模型"
    )
    vision_embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.SENTENCE_TRANSFORMERS_MINILM,
        description="视觉嵌入模型"
    )
    embedding_dimension: int = Field(
        default=384,
        description="嵌入向量维度"
    )
    
    # 文档处理配置
    chunk_size: int = Field(
        default=1000,
        description="文档分块大小"
    )
    chunk_overlap: int = Field(
        default=200,
        description="分块重叠大小"
    )
    max_image_size: int = Field(
        default=1024,
        description="最大图像尺寸(像素)"
    )
    
    # 检索配置
    retrieval_top_k: int = Field(
        default=10,
        description="检索结果数量"
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="相似度阈值"
    )
    rerank_enabled: bool = Field(
        default=True,
        description="是否启用重排序"
    )
    
    # 性能配置
    batch_size: int = Field(
        default=32,
        description="批处理大小"
    )
    cache_enabled: bool = Field(
        default=True,
        description="是否启用缓存"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="缓存过期时间(秒)"
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="最大并发请求数"
    )
    
class ProcessedDocument(BaseModel):
    """处理后的文档"""
    
    doc_id: str = Field(description="文档ID")
    source_file: str = Field(description="源文件路径")
    content_type: str = Field(description="内容类型")
    
    # 文本内容
    texts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文本块列表"
    )
    
    # 表格内容
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="表格列表"
    )
    
    # 图像内容
    images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="图像列表"
    )
    
    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="文档元数据"
    )
    
    # 摘要信息
    summary: Optional[str] = Field(
        default=None,
        description="文档摘要"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="关键词列表"
    )

class QueryContext(BaseModel):
    """查询上下文"""
    
    query: str = Field(description="查询文本")
    query_type: QueryType = Field(
        default=QueryType.TEXT,
        description="查询类型"
    )
    
    # 输入文件
    input_files: List[str] = Field(
        default_factory=list,
        description="输入文件列表"
    )
    
    # 查询参数
    requires_image_search: bool = Field(
        default=False,
        description="是否需要图像搜索"
    )
    requires_table_search: bool = Field(
        default=False,
        description="是否需要表格搜索"
    )
    
    # 过滤条件
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="过滤条件"
    )
    
    # 检索参数
    top_k: Optional[int] = Field(
        default=None,
        description="覆盖默认的top_k"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="覆盖默认的相似度阈值"
    )

class RetrievalResults(BaseModel):
    """检索结果"""
    
    # 文本结果
    texts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文本检索结果"
    )
    
    # 图像结果
    images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="图像检索结果"
    )
    
    # 表格结果
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="表格检索结果"
    )
    
    # 来源信息
    sources: List[str] = Field(
        default_factory=list,
        description="来源文件列表"
    )
    
    # 性能指标
    retrieval_time_ms: float = Field(
        default=0.0,
        description="检索时间(毫秒)"
    )
    total_results: int = Field(
        default=0,
        description="总结果数"
    )

class MultimodalContext(BaseModel):
    """多模态上下文"""
    
    texts: str = Field(
        default="",
        description="格式化的文本上下文"
    )
    images: List[str] = Field(
        default_factory=list,
        description="Base64编码的图像列表"
    )
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="格式化的表格数据"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="上下文元数据"
    )

class QAResponse(BaseModel):
    """问答响应"""
    
    answer: str = Field(description="回答内容")
    sources: List[str] = Field(
        default_factory=list,
        description="引用来源"
    )
    confidence: float = Field(
        default=0.0,
        description="置信度分数"
    )
    processing_time: float = Field(
        default=0.0,
        description="处理时间(秒)"
    )
    context_used: Dict[str, int] = Field(
        default_factory=dict,
        description="使用的上下文统计"
    )

class RetrievalMetrics(BaseModel):
    """检索性能指标"""
    
    query_count: int = Field(default=0)
    avg_retrieval_time_ms: float = Field(default=0.0)
    avg_similarity_score: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0)
    memory_usage_mb: float = Field(default=0.0)
    vector_index_size: int = Field(default=0)

class QualityMetrics(BaseModel):
    """质量评估指标"""
    
    retrieval_precision: float = Field(default=0.0)
    retrieval_recall: float = Field(default=0.0)
    retrieval_f1: float = Field(default=0.0)
    image_understanding_accuracy: float = Field(default=0.0)
    document_parsing_success_rate: float = Field(default=0.0)
    end_to_end_quality_score: float = Field(default=0.0)
