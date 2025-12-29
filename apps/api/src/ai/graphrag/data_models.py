"""
GraphRAG核心数据模型

定义GraphRAG系统中使用的所有核心数据结构，包括：
- 查询类型和检索模式
- 图谱上下文和推理路径
- 知识源和融合结果
- 请求响应格式
"""

from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from dataclasses import dataclass
import uuid

class QueryType(str, Enum):
    """查询类型枚举"""
    SIMPLE = "simple"
    MULTI_ENTITY = "multi_entity"
    RELATIONAL = "relational"
    COMPLEX_REASONING = "complex_reasoning"
    COMPOSITIONAL = "compositional"

class RetrievalMode(str, Enum):
    """检索模式枚举"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class GraphContext:
    """图谱上下文"""
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    subgraph: Dict[str, Any]
    reasoning_paths: List[Dict[str, Any]]
    expansion_depth: int
    confidence_score: float

    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.entities, list):
            self.entities = []
        if not isinstance(self.relations, list):
            self.relations = []
        if not isinstance(self.subgraph, dict):
            self.subgraph = {}
        if not isinstance(self.reasoning_paths, list):
            self.reasoning_paths = []
        
        # 确保置信度在有效范围内
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entities": self.entities,
            "relations": self.relations,
            "subgraph": self.subgraph,
            "reasoning_paths": self.reasoning_paths,
            "expansion_depth": self.expansion_depth,
            "confidence_score": self.confidence_score
        }

@dataclass
class ReasoningPath:
    """推理路径"""
    path_id: str
    entities: List[str]
    relations: List[str]
    path_score: float
    explanation: str
    evidence: List[Dict[str, Any]]
    hops_count: int

    def __post_init__(self):
        """初始化后处理"""
        if not self.path_id:
            self.path_id = str(uuid.uuid4())
        if not isinstance(self.entities, list):
            self.entities = []
        if not isinstance(self.relations, list):
            self.relations = []
        if not isinstance(self.evidence, list):
            self.evidence = []
        
        # 确保路径评分在有效范围内
        self.path_score = max(0.0, min(1.0, self.path_score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "path_id": self.path_id,
            "entities": self.entities,
            "relations": self.relations,
            "path_score": self.path_score,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "hops_count": self.hops_count
        }

@dataclass
class KnowledgeSource:
    """知识源"""
    source_type: str  # vector, graph, reasoning
    content: str
    confidence: float
    metadata: Dict[str, Any]
    graph_context: Optional[GraphContext] = None

    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # 确保置信度在有效范围内
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source_type": self.source_type,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "graph_context": self.graph_context.to_dict() if self.graph_context else None
        }

class GraphRAGRequest(TypedDict):
    """GraphRAG请求"""
    query: str
    retrieval_mode: RetrievalMode
    max_docs: int
    include_reasoning: bool
    expansion_depth: int
    confidence_threshold: float
    query_type: Optional[QueryType]
    filters: Optional[Dict[str, Any]]

class GraphRAGResponse(TypedDict):
    """GraphRAG响应"""
    query_id: str
    query: str
    documents: List[Dict[str, Any]]
    graph_context: Dict[str, Any]  # GraphContext.to_dict()
    reasoning_results: List[Dict[str, Any]]  # List[ReasoningPath.to_dict()]
    knowledge_sources: List[Dict[str, Any]]  # List[KnowledgeSource.to_dict()]
    fusion_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: str

@dataclass
class QueryDecomposition:
    """查询分解结果"""
    original_query: str
    sub_queries: List[str]
    entity_queries: List[Dict[str, Any]]
    relation_queries: List[Dict[str, Any]]
    decomposition_strategy: str
    complexity_score: float

    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.sub_queries, list):
            self.sub_queries = []
        if not isinstance(self.entity_queries, list):
            self.entity_queries = []
        if not isinstance(self.relation_queries, list):
            self.relation_queries = []
        
        # 确保复杂度评分在有效范围内
        self.complexity_score = max(0.0, min(1.0, self.complexity_score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_query": self.original_query,
            "sub_queries": self.sub_queries,
            "entity_queries": self.entity_queries,
            "relation_queries": self.relation_queries,
            "decomposition_strategy": self.decomposition_strategy,
            "complexity_score": self.complexity_score
        }

class FusionResult(TypedDict):
    """融合结果"""
    final_ranking: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    conflicts_detected: List[Dict[str, Any]]
    resolution_strategy: str
    consistency_score: float

@dataclass
class EntityRecognitionResult:
    """实体识别结果"""
    text: str
    canonical_form: Optional[str]
    entity_type: Optional[str] 
    confidence: float
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]

    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # 确保置信度在有效范围内
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "canonical_form": self.canonical_form,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata
        }

@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    max_expansion_depth: int = 3
    reasoning_timeout: float = 5.0
    cache_ttl: int = 3600
    confidence_threshold: float = 0.6
    max_reasoning_paths: int = 10
    vector_search_limit: int = 20
    graph_traversal_limit: int = 100
    fusion_strategy: str = "weighted_evidence"
    enable_caching: bool = True
    enable_reasoning: bool = True
    enable_conflict_resolution: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_expansion_depth": self.max_expansion_depth,
            "reasoning_timeout": self.reasoning_timeout,
            "cache_ttl": self.cache_ttl,
            "confidence_threshold": self.confidence_threshold,
            "max_reasoning_paths": self.max_reasoning_paths,
            "vector_search_limit": self.vector_search_limit,
            "graph_traversal_limit": self.graph_traversal_limit,
            "fusion_strategy": self.fusion_strategy,
            "enable_caching": self.enable_caching,
            "enable_reasoning": self.enable_reasoning,
            "enable_conflict_resolution": self.enable_conflict_resolution
        }

# 辅助函数
def create_graph_rag_request(
    query: str,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    max_docs: int = 10,
    include_reasoning: bool = True,
    expansion_depth: int = 2,
    confidence_threshold: float = 0.6,
    query_type: Optional[QueryType] = None,
    filters: Optional[Dict[str, Any]] = None
) -> GraphRAGRequest:
    """创建GraphRAG请求"""
    return GraphRAGRequest(
        query=query,
        retrieval_mode=retrieval_mode,
        max_docs=max_docs,
        include_reasoning=include_reasoning,
        expansion_depth=expansion_depth,
        confidence_threshold=confidence_threshold,
        query_type=query_type,
        filters=filters or {}
    )

def create_empty_graph_context() -> GraphContext:
    """创建空的图谱上下文"""
    return GraphContext(
        entities=[],
        relations=[],
        subgraph={},
        reasoning_paths=[],
        expansion_depth=0,
        confidence_score=0.0
    )

def create_empty_reasoning_path() -> ReasoningPath:
    """创建空的推理路径"""
    return ReasoningPath(
        path_id=str(uuid.uuid4()),
        entities=[],
        relations=[],
        path_score=0.0,
        explanation="",
        evidence=[],
        hops_count=0
    )

def validate_graph_rag_request(request: GraphRAGRequest) -> List[str]:
    """验证GraphRAG请求的有效性"""
    errors = []
    
    if not request.get("query", "").strip():
        errors.append("查询不能为空")
    
    if request.get("max_docs", 0) <= 0:
        errors.append("max_docs必须大于0")
    
    if request.get("expansion_depth", 0) < 0:
        errors.append("expansion_depth不能为负数")
    
    confidence_threshold = request.get("confidence_threshold", 0)
    if not (0 <= confidence_threshold <= 1):
        errors.append("confidence_threshold必须在0-1之间")
    
    retrieval_mode = request.get("retrieval_mode")
    if retrieval_mode and retrieval_mode not in [mode.value for mode in RetrievalMode]:
        errors.append(f"无效的检索模式: {retrieval_mode}")
    
    return errors
