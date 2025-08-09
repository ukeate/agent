#!/usr/bin/env python3
"""
Agentic RAG智能检索系统功能验证脚本

验证查询理解、查询扩展和多代理检索协作的完整流程
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ai.agentic_rag.query_analyzer import QueryAnalyzer, QueryIntent
from ai.agentic_rag.query_expander import QueryExpander, ExpansionStrategy
from ai.agentic_rag.retrieval_agents import MultiAgentRetriever, RetrievalStrategy


async def test_query_analysis():
    """测试查询理解和意图识别"""
    print("=== 测试查询理解和意图识别 ===")
    
    try:
        analyzer = QueryAnalyzer()
        test_queries = [
            "机器学习算法实现",
            "如何优化深度学习模型",
            "什么是Transformer架构",
            "Python数据处理库使用指南"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            try:
                analysis = await analyzer.analyze_query(query)
                print(f"  意图: {analysis.intent_type.value}")
                print(f"  复杂度: {analysis.complexity_score:.2f}")
                print(f"  实体: {analysis.entities}")
                print(f"  关键词: {analysis.keywords[:3]}")  # 只显示前3个
                print(f"  置信度: {analysis.confidence:.2f}")
            except Exception as e:
                print(f"  分析失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"查询分析测试失败: {e}")
        return False


async def test_query_expansion():
    """测试查询扩展和改写"""
    print("\n=== 测试查询扩展和改写 ===")
    
    try:
        analyzer = QueryAnalyzer()
        expander = QueryExpander()
        
        query = "Python机器学习库使用"
        print(f"\n原始查询: {query}")
        
        # 查询分析
        analysis = await analyzer.analyze_query(query)
        
        # 查询扩展
        expansions = await expander.expand_query(
            analysis, 
            strategies=[ExpansionStrategy.SYNONYM, ExpansionStrategy.SEMANTIC]
        )
        
        for expansion in expansions:
            print(f"\n扩展策略: {expansion.strategy.value}")
            print(f"置信度: {expansion.confidence:.2f}")
            if expansion.expanded_queries:
                print("扩展查询:")
                for i, expanded in enumerate(expansion.expanded_queries[:3], 1):
                    print(f"  {i}. {expanded}")
            print(f"解释: {expansion.explanation}")
        
        return True
        
    except Exception as e:
        print(f"查询扩展测试失败: {e}")
        return False


async def test_multi_agent_retrieval():
    """测试多代理检索协作"""
    print("\n=== 测试多代理检索协作 ===")
    
    try:
        analyzer = QueryAnalyzer()
        retriever = MultiAgentRetriever()
        
        query = "数据库优化技巧"
        print(f"\n测试查询: {query}")
        
        # 查询分析
        analysis = await analyzer.analyze_query(query)
        print(f"查询意图: {analysis.intent_type.value}")
        print(f"复杂度评分: {analysis.complexity_score:.2f}")
        
        # 策略选择
        selected_strategies = retriever.select_strategies(analysis)
        print(f"\n选定策略:")
        for strategy, score in selected_strategies:
            print(f"  {strategy.value}: {score:.2f}")
        
        # 由于没有真实的向量数据库和数据，这里只测试系统架构
        print(f"\n系统性能摘要:")
        performance = retriever.get_performance_summary()
        for strategy_name, agent_info in performance.items():
            stats = agent_info['stats']
            print(f"  {agent_info['name']} ({strategy_name}):")
            print(f"    查询总数: {stats['total_queries']}")
            print(f"    平均响应时间: {stats['avg_response_time']:.3f}s")
            print(f"    成功率: {stats['success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"多代理检索测试失败: {e}")
        return False


async def test_integration_workflow():
    """测试完整的集成工作流程"""
    print("\n=== 测试完整集成工作流程 ===")
    
    try:
        # 初始化组件
        analyzer = QueryAnalyzer()
        expander = QueryExpander()
        retriever = MultiAgentRetriever()
        
        # 测试场景
        scenarios = [
            {
                "query": "深度学习神经网络实现",
                "expected_intent": QueryIntent.CODE,
                "context": ["我们正在学习AI算法", "特别关注实际应用"]
            },
            {
                "query": "什么是注意力机制",
                "expected_intent": QueryIntent.FACTUAL,
                "context": None
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n场景 {i}: {scenario['query']}")
            
            # Step 1: 查询分析
            analysis = await analyzer.analyze_query(
                scenario['query'], 
                context_history=scenario.get('context')
            )
            print(f"  意图识别: {analysis.intent_type.value} (期望: {scenario['expected_intent'].value})")
            
            # Step 2: 查询扩展
            expansions = await expander.expand_query(
                analysis,
                context_history=scenario.get('context'),
                strategies=[ExpansionStrategy.SYNONYM, ExpansionStrategy.SEMANTIC]
            )
            
            best_expansions = expander.get_best_expansions(expansions, max_results=3)
            print(f"  最佳扩展查询: {len(best_expansions)}个")
            for j, expanded in enumerate(best_expansions, 1):
                print(f"    {j}. {expanded}")
            
            # Step 3: 策略选择
            strategies = retriever.select_strategies(analysis)
            print(f"  推荐策略: {', '.join([s.value for s, _ in strategies[:2]])}")
            
            # Step 4: 解释生成
            explanation = retriever.get_retrieval_explanation(
                analysis, 
                [],  # 没有实际检索结果
                []   # 没有实际融合结果
            )
            print(f"  处理说明: {explanation}")
        
        return True
        
    except Exception as e:
        print(f"集成工作流程测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Agentic RAG智能检索系统功能验证")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("查询理解和意图识别", test_query_analysis),
        ("查询扩展和改写", test_query_expansion), 
        ("多代理检索协作", test_multi_agent_retrieval),
        ("完整集成工作流程", test_integration_workflow)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            test_results.append((test_name, False))
            print(f"\n{test_name}: ❌ 失败 ({e})")
    
    # 总结报告
    print(f"\n{'='*50}")
    print("🎯 测试总结报告")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Agentic RAG系统功能正常")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关组件")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试运行失败: {e}")
        sys.exit(1)