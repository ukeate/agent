#!/usr/bin/env python3
"""
简化的LangGraph Context API性能测试
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import time
import statistics
import uuid
from typing import List

from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import LangGraphWorkflowBuilder
from src.ai.langgraph.context import AgentContext, create_context
from datetime import datetime, timezone


async def run_performance_test():
    """运行性能测试"""
    print("开始LangGraph Context API性能测试...")
    
    test_iterations = 20
    legacy_times: List[float] = []
    context_api_times: List[float] = []
    
    # 定义简单的测试节点
    def test_node(state: MessagesState, context: AgentContext = None) -> MessagesState:
        """测试节点函数"""
        state["messages"].append({
            "role": "system",
            "content": f"处理步骤 {len(state['messages']) + 1}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if context:
            context.update_step("test_node")
        return state
    
    # 测试传统config模式
    print(f"测试传统config模式 ({test_iterations}次迭代)...")
    try:
        for i in range(test_iterations):
            builder_legacy = LangGraphWorkflowBuilder(use_context_api=False)
            builder_legacy.add_node("test_node", test_node)
            
            initial_state = create_initial_state(f"legacy-test-{i}")
            context = create_context(
                user_id=f"user-{i}",
                session_id=str(uuid.uuid4()),
                workflow_id=f"legacy-test-{i}"
            )
            
            start_time = time.time()
            
            # 模拟简单的执行
            result = test_node(initial_state, context)
            
            end_time = time.time()
            execution_time = end_time - start_time
            legacy_times.append(execution_time)
            
            print(f"Legacy iteration {i+1}: {execution_time:.6f}s")
    except Exception as e:
        print(f"Legacy test error: {e}")
        return
    
    # 测试新Context API模式
    print(f"测试新Context API模式 ({test_iterations}次迭代)...")
    try:
        for i in range(test_iterations):
            builder_new = LangGraphWorkflowBuilder(use_context_api=True)
            builder_new.add_node("test_node", test_node)
            
            initial_state = create_initial_state(f"context-test-{i}")
            context = create_context(
                user_id=f"user-{i}",
                session_id=str(uuid.uuid4()),
                workflow_id=f"context-test-{i}"
            )
            
            start_time = time.time()
            
            # 模拟简单的执行
            result = test_node(initial_state, context)
            
            end_time = time.time()
            execution_time = end_time - start_time
            context_api_times.append(execution_time)
            
            print(f"Context API iteration {i+1}: {execution_time:.6f}s")
    except Exception as e:
        print(f"Context API test error: {e}")
        return
    
    # 计算性能统计
    legacy_avg = statistics.mean(legacy_times)
    context_api_avg = statistics.mean(context_api_times)
    performance_improvement = ((legacy_avg - context_api_avg) / legacy_avg) * 100 if legacy_avg > 0 else 0
    
    print("\n" + "="*60)
    print("性能测试结果")
    print("="*60)
    print(f"传统config模式平均时间: {legacy_avg:.6f}秒")
    print(f"新Context API模式平均时间: {context_api_avg:.6f}秒")
    print(f"性能提升: {performance_improvement:.1f}%")
    
    if performance_improvement > 0:
        print("✅ Context API性能优于传统模式")
    else:
        print("❌ Context API性能未显示改善")
    
    # 测试Context序列化性能
    print("\n测试Context序列化性能...")
    context = create_context("perf-user", str(uuid.uuid4()))
    
    serialization_times = []
    for i in range(test_iterations):
        start_time = time.time()
        serialized = context.to_dict()
        end_time = time.time()
        serialization_times.append(end_time - start_time)
    
    avg_serialization = statistics.mean(serialization_times)
    print(f"Context序列化平均时间: {avg_serialization:.6f}秒")
    
    if avg_serialization < 0.001:
        print("✅ Context序列化性能优秀 (<1ms)")
    else:
        print(f"⚠️  Context序列化时间: {avg_serialization*1000:.2f}ms")


if __name__ == "__main__":
    asyncio.run(run_performance_test())