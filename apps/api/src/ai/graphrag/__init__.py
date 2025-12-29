"""
GraphRAG (图谱增强检索生成) 系统

结合知识图谱与向量检索的混合RAG系统，提供：
- 图谱增强的检索策略
- 实体关系上下文扩展  
- 图谱引导的问题分解
- 多源知识融合算法
- 推理路径生成和评分
"""

from .data_models import (
    QueryType,
    RetrievalMode,
    GraphContext,
    ReasoningPath,
    KnowledgeSource,
    GraphRAGRequest,
    GraphRAGResponse,
    QueryDecomposition,
    FusionResult
)

# 延迟导入核心组件，避免循环依赖
def get_graphrag_engine():
    """获取GraphRAG核心引擎（延迟导入）"""
    from .core_engine import get_graphrag_engine
    return get_graphrag_engine()

def get_query_analyzer():
    """获取查询分析器（延迟导入）"""
    from .query_analyzer import QueryAnalyzer
    return QueryAnalyzer()

def get_knowledge_fusion():
    """获取知识融合器（延迟导入）"""
    from .knowledge_fusion import KnowledgeFusion
    return KnowledgeFusion()

def get_reasoning_engine():
    """获取推理引擎（延迟导入）"""
    from .reasoning_engine import ReasoningEngine
    return ReasoningEngine()

def get_cache_manager():
    """获取缓存管理器（延迟导入）"""
    from .cache_manager import CacheManager
    return CacheManager()

__all__ = [
    # Data models
    "QueryType",
    "RetrievalMode", 
    "GraphContext",
    "ReasoningPath",
    "KnowledgeSource",
    "GraphRAGRequest",
    "GraphRAGResponse",
    "QueryDecomposition",
    "FusionResult",
    
    # Factory functions
    "get_graphrag_engine",
    "get_query_analyzer",
    "get_knowledge_fusion",
    "get_reasoning_engine",
    "get_cache_manager"
]
