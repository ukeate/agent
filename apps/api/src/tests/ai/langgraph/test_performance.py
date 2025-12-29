"""
性能基准测试
对比LangGraph v0.6 Context API升级前后的性能
"""

import pytest
import time
import asyncio
from typing import List, Dict, Any
from statistics import mean, stdev
from src.ai.langgraph.context import AgentContext, create_default_context
from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import (

    LangGraphWorkflowBuilder,
    create_simple_workflow,
    create_conditional_workflow
)

from src.core.logging import get_logger
logger = get_logger(__name__)

class TestPerformanceBenchmark:
    """性能基准测试"""
    
    @pytest.mark.asyncio
    async def test_simple_workflow_performance(self):
        """测试简单工作流性能"""
        execution_times = []
        num_runs = 10
        
        for i in range(num_runs):
            builder = create_simple_workflow()
            initial_state = create_initial_state(f"perf-test-{i}")
            context = create_default_context(
                user_id=f"perf_user_{i}",
                session_id=f"perf_session_{i}"
            )
            
            start_time = time.perf_counter()
            result = await builder.execute(initial_state, context)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # 转换为毫秒
            execution_times.append(execution_time)
            
            assert result["metadata"]["status"] == "completed"
        
        # 计算统计数据
        avg_time = mean(execution_times)
        std_time = stdev(execution_times) if len(execution_times) > 1 else 0
        
        logger.info(f"\n简单工作流性能测试结果:")
        logger.info(f"  运行次数: {num_runs}")
        logger.info(f"  平均执行时间: {avg_time:.2f}ms")
        logger.info(f"  标准差: {std_time:.2f}ms")
        logger.info(f"  最小时间: {min(execution_times):.2f}ms")
        logger.info(f"  最大时间: {max(execution_times):.2f}ms")
        
        # 性能断言 - 平均执行时间应该小于100ms
        assert avg_time < 100, f"平均执行时间 {avg_time:.2f}ms 超过预期"
    
    @pytest.mark.asyncio
    async def test_conditional_workflow_performance(self):
        """测试条件工作流性能"""
        execution_times = []
        num_runs = 10
        
        for i in range(num_runs):
            builder = create_conditional_workflow()
            initial_state = create_initial_state(f"cond-perf-test-{i}")
            context = create_default_context(
                user_id=f"perf_user_{i}",
                session_id=f"perf_session_{i}"
            )
            
            start_time = time.perf_counter()
            result = await builder.execute(initial_state, context)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # 转换为毫秒
            execution_times.append(execution_time)
            
            assert result["metadata"]["status"] == "completed"
        
        # 计算统计数据
        avg_time = mean(execution_times)
        std_time = stdev(execution_times) if len(execution_times) > 1 else 0
        
        logger.info(f"\n条件工作流性能测试结果:")
        logger.info(f"  运行次数: {num_runs}")
        logger.info(f"  平均执行时间: {avg_time:.2f}ms")
        logger.info(f"  标准差: {std_time:.2f}ms")
        logger.info(f"  最小时间: {min(execution_times):.2f}ms")
        logger.info(f"  最大时间: {max(execution_times):.2f}ms")
        
        # 性能断言 - 平均执行时间应该小于150ms
        assert avg_time < 150, f"平均执行时间 {avg_time:.2f}ms 超过预期"
    
    @pytest.mark.asyncio
    async def test_context_creation_performance(self):
        """测试上下文创建性能"""
        creation_times = []
        num_runs = 1000
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            context = AgentContext(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                conversation_id=f"conv_{i}",
                agent_id=f"agent_{i}",
                workflow_id=f"workflow_{i}",
                metadata={"index": i}
            )
            _ = context.to_dict()
            end_time = time.perf_counter()
            
            creation_time = (end_time - start_time) * 1000  # 转换为毫秒
            creation_times.append(creation_time)
        
        # 计算统计数据
        avg_time = mean(creation_times)
        std_time = stdev(creation_times) if len(creation_times) > 1 else 0
        
        logger.info(f"\n上下文创建性能测试结果:")
        logger.info(f"  运行次数: {num_runs}")
        logger.info(f"  平均创建时间: {avg_time:.4f}ms")
        logger.info(f"  标准差: {std_time:.4f}ms")
        logger.info(f"  最小时间: {min(creation_times):.4f}ms")
        logger.info(f"  最大时间: {max(creation_times):.4f}ms")
        
        # 性能断言 - 平均创建时间应该小于0.1ms
        assert avg_time < 0.1, f"平均创建时间 {avg_time:.4f}ms 超过预期"
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self):
        """测试并发工作流性能"""
        num_concurrent = 5
        
        async def run_workflow(index: int) -> float:
            builder = create_simple_workflow()
            initial_state = create_initial_state(f"concurrent-{index}")
            context = create_default_context(
                user_id=f"concurrent_user_{index}",
                session_id=f"concurrent_session_{index}"
            )
            
            start_time = time.perf_counter()
            result = await builder.execute(initial_state, context)
            end_time = time.perf_counter()
            
            assert result["metadata"]["status"] == "completed"
            return (end_time - start_time) * 1000
        
        # 并发执行
        start_time = time.perf_counter()
        tasks = [run_workflow(i) for i in range(num_concurrent)]
        execution_times = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # 计算统计数据
        avg_time = mean(execution_times)
        
        logger.info(f"\n并发工作流性能测试结果:")
        logger.info(f"  并发数: {num_concurrent}")
        logger.info(f"  总执行时间: {total_time:.2f}ms")
        logger.info(f"  平均单个执行时间: {avg_time:.2f}ms")
        logger.info(f"  并发效率: {(sum(execution_times) / total_time * 100):.1f}%")
        
        # 性能断言 - 并发执行应该比串行快
        serial_time = sum(execution_times)
        assert total_time < serial_time * 0.8, "并发执行效率低于预期"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """测试内存效率（简单测试）"""
        import gc
        import sys
        
        # 强制垃圾回收
        gc.collect()
        
        contexts = []
        for i in range(100):
            context = AgentContext(
                user_id=f"mem_user_{i}",
                session_id=f"mem_session_{i}",
                metadata={"data": "x" * 100}  # 添加一些数据
            )
            contexts.append(context)
        
        # 计算对象大小（粗略估计）
        total_size = sum(sys.getsizeof(ctx.to_dict()) for ctx in contexts)
        avg_size = total_size / len(contexts)
        
        logger.info(f"\n内存效率测试结果:")
        logger.info(f"  创建上下文数: {len(contexts)}")
        logger.info(f"  平均对象大小: {avg_size:.0f} bytes")
        logger.info(f"  总内存使用: {total_size / 1024:.2f} KB")
        
        # 内存断言 - 每个上下文应该小于2KB
        assert avg_size < 2048, f"平均对象大小 {avg_size:.0f} bytes 超过预期"
