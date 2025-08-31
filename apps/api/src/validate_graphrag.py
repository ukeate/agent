#!/usr/bin/env python3
"""
GraphRAG系统验证脚本

验证GraphRAG组件的基本功能和集成
"""

import sys
import traceback
from typing import Dict, Any

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
        print("✓ GraphRAG请求创建成功")
        
        # 测试请求验证
        errors = validate_graph_rag_request(request)
        print(f"✓ 请求验证完成，错误数: {len(errors)}")
        
        # 测试空上下文创建
        empty_context = create_empty_graph_context()
        print(f"✓ 空上下文创建成功，实体数: {len(empty_context.entities)}")
        
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
        print("✓ GraphContext创建和序列化成功")
        
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
        print("✓ 推理路径创建和序列化成功")
        
        # 测试知识源
        source = KnowledgeSource(
            source_type="vector",
            content="测试内容",
            confidence=0.8,
            metadata={"source": "test"}
        )
        source_dict = source.to_dict()
        print("✓ 知识源创建和序列化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据模型验证失败: {e}")
        traceback.print_exc()
        return False

def validate_cache_manager():
    """验证缓存管理器"""
    try:
        from ai.graphrag.cache_manager import CacheManager
        
        # 创建缓存管理器实例
        cache_manager = CacheManager()
        print("✓ 缓存管理器创建成功")
        
        # 测试缓存键生成
        cache_key = cache_manager._generate_cache_key("测试查询", "hybrid", {"param": "value"})
        print(f"✓ 缓存键生成成功: {cache_key[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 缓存管理器验证失败: {e}")
        traceback.print_exc()
        return False

def validate_query_analyzer():
    """验证查询分析器"""
    try:
        from ai.graphrag.query_analyzer import QueryAnalyzer
        
        # 创建查询分析器实例
        analyzer = QueryAnalyzer()
        print("✓ 查询分析器创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 查询分析器验证失败: {e}")
        traceback.print_exc()
        return False

def validate_knowledge_fusion():
    """验证知识融合器"""
    try:
        from ai.graphrag.knowledge_fusion import KnowledgeFusion
        
        # 创建知识融合器实例
        fusion = KnowledgeFusion()
        print("✓ 知识融合器创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 知识融合器验证失败: {e}")
        traceback.print_exc()
        return False

def validate_reasoning_engine():
    """验证推理引擎"""
    try:
        from ai.graphrag.reasoning_engine import ReasoningEngine
        
        # 创建推理引擎实例
        engine = ReasoningEngine()
        print("✓ 推理引擎创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 推理引擎验证失败: {e}")
        traceback.print_exc()
        return False

def validate_core_engine():
    """验证核心引擎"""
    try:
        from ai.graphrag.core_engine import GraphRAGEngine
        
        # 创建核心引擎实例
        engine = GraphRAGEngine()
        print("✓ 核心引擎创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 核心引擎验证失败: {e}")
        traceback.print_exc()
        return False

def validate_api_integration():
    """验证API集成"""
    try:
        # 验证GraphRAG API模块导入
        from api.v1.graphrag import router as graphrag_router
        print("✓ GraphRAG API路由导入成功")
        
        # 验证RAG集成模块更新
        from api.v1.rag import router as rag_router
        print("✓ RAG API路由导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ API集成验证失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("开始GraphRAG系统验证...")
    print("=" * 50)
    
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
        print(f"\n验证 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} 验证出现异常: {e}")
            results[test_name] = False
    
    # 生成验证报告
    print("\n" + "=" * 50)
    print("GraphRAG系统验证报告")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:<15} : {status}")
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 GraphRAG系统验证完全成功!")
        return 0
    else:
        print("⚠️  GraphRAG系统存在问题，需要修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())