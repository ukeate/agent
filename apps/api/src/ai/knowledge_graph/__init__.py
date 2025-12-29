"""
知识图谱构建模块

提供实体识别、关系抽取、实体链接等核心功能
支持多语言、多模型集成和高性能批处理
"""

from .data_models import (
    Entity, Relation, EntityType, RelationType,
    KnowledgeGraph, TripleStore,
    EntityModel, RelationModel,
    ExtractionRequest, ExtractionResponse,
    BatchProcessingResult, BatchProcessingRequest, BatchProcessingResponse
)
from .entity_recognizer import MultiModelEntityRecognizer
from .relation_extractor import RelationExtractor
from .entity_linker import EntityLinker
from .multilingual_processor import MultilingualProcessor
from .batch_processor import (
    BatchProcessor, DistributedBatchProcessor,
    BatchConfig, BatchTask, ProcessingMetrics,
    ProcessingStatus, CacheStrategy
)
from .knowledge_extraction import router as knowledge_router

__all__ = [
    # 核心数据结构
    "Entity",
    "Relation",
    "EntityType",
    "RelationType",
    "KnowledgeGraph",
    "TripleStore",
    
    # API模型
    "EntityModel",
    "RelationModel",
    "ExtractionRequest",
    "ExtractionResponse",
    "BatchProcessingResult",
    "BatchProcessingRequest", 
    "BatchProcessingResponse",
    
    # 处理器
    "MultiModelEntityRecognizer",
    "RelationExtractor",
    "EntityLinker",
    "MultilingualProcessor",
    
    # 批处理
    "BatchProcessor",
    "DistributedBatchProcessor",
    "BatchConfig",
    "BatchTask",
    "ProcessingMetrics",
    "ProcessingStatus",
    "CacheStrategy",
    
    # API路由
    "knowledge_router"
]
