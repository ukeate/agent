import sys
from typing import Dict, Any
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
GraphRAG系统验证脚本

验证GraphRAG组件的基本功能和集成
"""

def validate_data_models():
    """验证数据模型"""
    try:
        from ai.graphrag.data_models import (
            GraphRAGRequest,
            GraphContext,
            ReasoningPath,
            KnowledgeSource,
            create_graph_rag_request,
            create_empty_graph_context,
            validate_graph_rag_request,
            RetrievalMode
        )
        
        # 测试创建GraphRAG请求
        request = create_graph_rag_request(
            query="测试查询",
            retrieval_mode=RetrievalMode.HYBRID,
            max_docs=10
        )
        logger.info("GraphRAG请求创建成功")
        
        # 测试请求验证
        errors = validate_graph_rag_request(request)
        logger.info("请求验证完成", error_count=len(errors))
        
        # 测试空上下文创建
        empty_context = create_empty_graph_context()
        logger.info("空上下文创建成功", entity_count=len(empty_context.entities))
        
        # 测试GraphContext创建
        context = GraphContext(
            entities=[{"id": "1", "name": "测试实体"}],
            relations=[{"type": "RELATED", "source": "1", "target": "2"}],
            subgraph={},
            reasoning_paths=[],
            expansion_depth=1,
            confidence_score=0.8
        )
        context_dict = context.to_dict()
        logger.info("GraphContext创建和序列化成功")
        
        # 测试推理路径
        path = ReasoningPath(
            path_id="test",
            entities=["实体1", "实体2"],
            relations=["关系1"],
            path_score=0.9,
            explanation="测试推理",
            evidence=[{"fact": "事实"}],
            hops_count=1
        )
        path_dict = path.to_dict()
        logger.info("推理路径创建和序列化成功")
        
        # 测试知识源
        source = KnowledgeSource(
            source_type="vector",
            content="测试内容",
            confidence=0.8,
            metadata={"source": "test"}
        )
        source_dict = source.to_dict()
        logger.info("知识源创建和序列化成功")
        
        return True
        
    except Exception:
        logger.exception("数据模型验证失败")
        return False

def validate_cache_manager():
    """验证缓存管理器"""
    try:
        from ai.graphrag.cache_manager import CacheManager
        
        # 创建缓存管理器实例
        cache_manager = CacheManager()
        logger.info("缓存管理器创建成功")
        
        # 测试缓存键生成
        cache_key = cache_manager._generate_cache_key("测试查询", "hybrid", {"param": "value"})
        logger.info("缓存键生成成功", cache_key_prefix=cache_key[:50])
        
        return True
        
    except Exception:
        logger.exception("缓存管理器验证失败")
        return False

def validate_query_analyzer():
    """验证查询分析器"""
    try:
        from ai.graphrag.query_analyzer import QueryAnalyzer
        
        # 创建查询分析器实例
        analyzer = QueryAnalyzer()
        logger.info("查询分析器创建成功")
        
        return True
        
    except Exception:
        logger.exception("查询分析器验证失败")
        return False

def validate_knowledge_fusion():
    """验证知识融合器"""
    try:
        from ai.graphrag.knowledge_fusion import KnowledgeFusion
        
        # 创建知识融合器实例
        fusion = KnowledgeFusion()
        logger.info("知识融合器创建成功")
        
        return True
        
    except Exception:
        logger.exception("知识融合器验证失败")
        return False

def validate_reasoning_engine():
    """验证推理引擎"""
    try:
        from ai.graphrag.reasoning_engine import ReasoningEngine
        
        # 创建推理引擎实例
        engine = ReasoningEngine()
        logger.info("推理引擎创建成功")
        
        return True
        
    except Exception:
        logger.exception("推理引擎验证失败")
        return False

def validate_core_engine():
    """验证核心引擎"""
    try:
        from ai.graphrag.core_engine import GraphRAGEngine
        
        # 创建核心引擎实例
        engine = GraphRAGEngine()
        logger.info("核心引擎创建成功")
        
        return True
        
    except Exception:
        logger.exception("核心引擎验证失败")
        return False

def validate_api_integration():
    """验证API集成"""
    try:
        # 验证GraphRAG API模块导入
        from api.v1.graphrag import router as graphrag_router
        logger.info("GraphRAG API路由导入成功")
        
        # 验证RAG集成模块更新
        from api.v1.rag import router as rag_router
        logger.info("RAG API路由导入成功")
        
        return True
        
    except Exception:
        logger.exception("API集成验证失败")
        return False

def main():
    """主验证函数"""
    logger.info("开始GraphRAG系统验证")
    logger.info("验证分隔线", line="=" * 50)
    
    # 运行所有验证测试
    tests = [
        ("数据模型", validate_data_models),
        ("缓存管理器", validate_cache_manager),
        ("查询分析器", validate_query_analyzer),
        ("知识融合器", validate_knowledge_fusion),
        ("推理引擎", validate_reasoning_engine),
        ("核心引擎", validate_core_engine),
        ("API集成", validate_api_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info("开始验证", test_name=test_name)
        try:
            results[test_name] = test_func()
        except Exception:
            logger.exception("验证出现异常", test_name=test_name)
            results[test_name] = False
    
    # 生成验证报告
    logger.info("验证分隔线", line="=" * 50)
    logger.info("GraphRAG系统验证报告")
    logger.info("验证分隔线", line="=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        logger.info("验证结果", test_name=test_name, status=status)
    
    logger.info("总体结果", passed=passed, total=total)
    
    if passed == total:
        logger.info("GraphRAG系统验证完全成功")
        return 0
    else:
        logger.warning("GraphRAG系统存在问题，需要修复")
        return 1

if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
