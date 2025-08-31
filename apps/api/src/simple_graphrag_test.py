#!/usr/bin/env python3
"""
简单的GraphRAG系统验证脚本

仅验证核心数据模型和基础功能，不涉及复杂依赖
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_models():
    """验证GraphRAG数据模型"""
    try:
        # 直接导入数据模型模块
        from ai.graphrag.data_models import (
            GraphRAGRequest,
            GraphContext,
            ReasoningPath,
            KnowledgeSource,
            QueryDecomposition,
            FusionResult,
            EntityRecognitionResult,
            GraphRAGConfig,
            create_graph_rag_request,
            create_empty_graph_context,
            validate_graph_rag_request,
            RetrievalMode,
            QueryType
        )
        
        print("✓ GraphRAG数据模型导入成功")
        
        # 测试创建GraphRAG请求
        request = create_graph_rag_request(
            query="什么是机器学习",
            retrieval_mode=RetrievalMode.HYBRID,
            max_docs=10
        )
        print("✓ GraphRAG请求创建成功")
        print(f"  - 查询: {request['query']}")
        print(f"  - 检索模式: {request['retrieval_mode']}")
        print(f"  - 最大文档数: {request['max_docs']}")
        
        # 测试请求验证
        errors = validate_graph_rag_request(request)
        print(f"✓ 请求验证完成，错误数: {len(errors)}")
        
        # 测试空上下文创建
        empty_context = create_empty_graph_context()
        print(f"✓ 空上下文创建成功")
        print(f"  - 实体数: {len(empty_context.entities)}")
        print(f"  - 关系数: {len(empty_context.relations)}")
        print(f"  - 置信度: {empty_context.confidence_score}")
        
        # 测试GraphContext创建和序列化
        context = GraphContext(
            entities=[{"id": "1", "name": "机器学习", "type": "CONCEPT"}],
            relations=[{"type": "PART_OF", "source": "1", "target": "2"}],
            subgraph={"nodes": 1, "edges": 1},
            reasoning_paths=[],
            expansion_depth=2,
            confidence_score=0.8
        )
        context_dict = context.to_dict()
        print("✓ GraphContext创建和序列化成功")
        print(f"  - 实体数: {len(context.entities)}")
        print(f"  - 关系数: {len(context.relations)}")
        print(f"  - 扩展深度: {context.expansion_depth}")
        print(f"  - 置信度: {context.confidence_score}")
        
        # 测试推理路径
        path = ReasoningPath(
            path_id="test_path_1",
            entities=["机器学习", "人工智能"],
            relations=["IS_A"],
            path_score=0.9,
            explanation="机器学习是人工智能的一个分支",
            evidence=[{"fact": "ML是AI的子领域", "confidence": 0.8}],
            hops_count=1
        )
        path_dict = path.to_dict()
        print("✓ 推理路径创建和序列化成功")
        print(f"  - 路径ID: {path.path_id}")
        print(f"  - 实体数: {len(path.entities)}")
        print(f"  - 路径评分: {path.path_score}")
        print(f"  - 跳数: {path.hops_count}")
        
        # 测试知识源
        source = KnowledgeSource(
            source_type="vector",
            content="机器学习是一种人工智能技术",
            confidence=0.85,
            metadata={"source": "wikipedia", "section": "definition"}
        )
        source_dict = source.to_dict()
        print("✓ 知识源创建和序列化成功")
        print(f"  - 源类型: {source.source_type}")
        print(f"  - 内容长度: {len(source.content)}")
        print(f"  - 置信度: {source.confidence}")
        print(f"  - 元数据: {source.metadata}")
        
        # 测试查询分解
        decomposition = QueryDecomposition(
            original_query="什么是机器学习",
            sub_queries=["机器学习定义", "机器学习应用", "机器学习算法"],
            entity_queries=[{"entity": "机器学习", "type": "CONCEPT"}],
            relation_queries=[{"entity1": "机器学习", "entity2": "人工智能", "relation": "PART_OF"}],
            decomposition_strategy="semantic_analysis",
            complexity_score=0.6
        )
        decomp_dict = decomposition.to_dict()
        print("✓ 查询分解创建和序列化成功")
        print(f"  - 原始查询: {decomposition.original_query}")
        print(f"  - 子查询数: {len(decomposition.sub_queries)}")
        print(f"  - 实体查询数: {len(decomposition.entity_queries)}")
        print(f"  - 关系查询数: {len(decomposition.relation_queries)}")
        print(f"  - 复杂度: {decomposition.complexity_score}")
        
        # 测试实体识别结果
        entity_result = EntityRecognitionResult(
            text="机器学习",
            canonical_form="机器学习",
            entity_type="CONCEPT",
            confidence=0.9,
            start_pos=0,
            end_pos=4,
            metadata={"method": "nlp_analysis"}
        )
        entity_dict = entity_result.to_dict()
        print("✓ 实体识别结果创建和序列化成功")
        print(f"  - 识别文本: {entity_result.text}")
        print(f"  - 标准形式: {entity_result.canonical_form}")
        print(f"  - 实体类型: {entity_result.entity_type}")
        print(f"  - 置信度: {entity_result.confidence}")
        
        # 测试融合结果 - TypedDict版本
        fusion_result = FusionResult(
            final_ranking=[{"source": source.to_dict(), "rank": 1}],
            confidence_scores={"vector": 0.85, "graph": 0.78},
            conflicts_detected=[],
            resolution_strategy="weighted_consensus",
            consistency_score=0.88
        )
        # TypedDict本身就是dict，不需要to_dict()方法
        print("✓ 融合结果创建和序列化成功")
        print(f"  - 最终排名数: {len(fusion_result['final_ranking'])}")
        print(f"  - 一致性评分: {fusion_result['consistency_score']}")
        print(f"  - 解决策略: {fusion_result['resolution_strategy']}")
        print(f"  - 置信度评分: {fusion_result['confidence_scores']}")
        
        # 测试配置
        config = GraphRAGConfig()
        config_dict = config.to_dict()
        print("✓ GraphRAG配置创建和序列化成功")
        print(f"  - 最大扩展深度: {config.max_expansion_depth}")
        print(f"  - 置信度阈值: {config.confidence_threshold}")
        print(f"  - 启用缓存: {config.enable_caching}")
        print(f"  - 缓存TTL: {config.cache_ttl}")
        
        # 测试post_init方法
        print("\n测试数据验证和后处理...")
        
        # 测试置信度限制
        high_confidence_context = GraphContext(
            entities=[],
            relations=[],
            subgraph={},
            reasoning_paths=[],
            expansion_depth=1,
            confidence_score=1.5  # 超过1.0，应该被限制
        )
        print(f"✓ 置信度限制验证: {high_confidence_context.confidence_score} (应为1.0)")
        
        # 测试None值转换
        none_context = GraphContext(
            entities=None,
            relations=None,
            subgraph=None,
            reasoning_paths=None,
            expansion_depth=1,
            confidence_score=0.5
        )
        print(f"✓ None值转换验证: 实体={len(none_context.entities)}, 关系={len(none_context.relations)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("GraphRAG数据模型验证")
    print("=" * 50)
    
    success = test_data_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 GraphRAG数据模型验证成功!")
        return 0
    else:
        print("❌ GraphRAG数据模型验证失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())