"""
缓存性能测试
测试缓存系统在各种负载下的性能表现
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from src.ai.langgraph.caching import CacheConfig, create_node_cache
from src.ai.langgraph.context import AgentContext
from src.ai.langgraph.cached_node import cached_node
from src.ai.langgraph.cache_monitor import CacheMonitor


class TestCachePerformance:
    """缓存性能测试套件"""
    
    @pytest.fixture
    def performance_context(self):
        """性能测试上下文"""
        import uuid
        return AgentContext(
            user_id="perf_user",
            session_id="perf_session",
            workflow_id="perf_workflow",
            agent_id=str(uuid.uuid4()),
            session_context={}
        )
    
    @pytest.fixture
    async def performance_cache_setup(self):
        """性能测试缓存设置"""
        config = CacheConfig(
            backend="memory",
            max_entries=10000,
            ttl_default=3600,
            monitoring=True
        )
        cache = create_node_cache(config)
        monitor = CacheMonitor(cache)
        
        yield cache, monitor
        
        await cache.clear()
    
    @pytest.mark.asyncio
    async def test_cache_latency_benchmark(self, performance_cache_setup, performance_context):
        """缓存延迟基准测试"""
        cache, monitor = performance_cache_setup
        
        # 准备测试数据
        test_data = {
            "large_text": "x" * 10000,  # 10KB文本
            "complex_dict": {
                "nested": {
                    "data": [{"id": i, "value": f"item_{i}"} for i in range(100)]
                }
            },
            "timestamp": time.time()
        }
        
        # 测试设置操作延迟
        set_latencies = []
        for i in range(100):
            key = f"perf_test_{i}"
            start_time = time.time()
            await cache.set(key, test_data, ttl=3600)
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            set_latencies.append(latency)
        
        # 测试获取操作延迟
        get_latencies = []
        for i in range(100):
            key = f"perf_test_{i}"
            start_time = time.time()
            result = await cache.get(key)
            latency = (time.time() - start_time) * 1000
            get_latencies.append(latency)
            assert result == test_data
        
        # 分析延迟统计
        set_stats = {
            "mean": statistics.mean(set_latencies),
            "median": statistics.median(set_latencies),
            "p95": sorted(set_latencies)[int(len(set_latencies) * 0.95)],
            "p99": sorted(set_latencies)[int(len(set_latencies) * 0.99)],
            "max": max(set_latencies)
        }
        
        get_stats = {
            "mean": statistics.mean(get_latencies),
            "median": statistics.median(get_latencies),
            "p95": sorted(get_latencies)[int(len(get_latencies) * 0.95)],
            "p99": sorted(get_latencies)[int(len(get_latencies) * 0.99)],
            "max": max(get_latencies)
        }
        
        print(f"\n缓存性能基准测试结果:")
        print(f"SET操作延迟统计 (ms):")
        print(f"  平均: {set_stats['mean']:.2f}")
        print(f"  中位数: {set_stats['median']:.2f}")
        print(f"  P95: {set_stats['p95']:.2f}")
        print(f"  P99: {set_stats['p99']:.2f}")
        print(f"  最大: {set_stats['max']:.2f}")
        
        print(f"GET操作延迟统计 (ms):")
        print(f"  平均: {get_stats['mean']:.2f}")
        print(f"  中位数: {get_stats['median']:.2f}")
        print(f"  P95: {get_stats['p95']:.2f}")
        print(f"  P99: {get_stats['p99']:.2f}")
        print(f"  最大: {get_stats['max']:.2f}")
        
        # 性能断言
        assert set_stats['p95'] < 50.0  # 95%的SET操作应该在50ms内
        assert get_stats['p95'] < 20.0  # 95%的GET操作应该在20ms内
    
    @pytest.mark.asyncio
    async def test_cache_throughput_benchmark(self, performance_cache_setup, performance_context):
        """缓存吞吐量基准测试"""
        cache, monitor = performance_cache_setup
        
        # 创建缓存节点用于测试
        @cached_node(name="throughput_test", cache=cache)
        async def compute_intensive_task(context: AgentContext, task_id: int, complexity: int):
            # 模拟计算密集型任务
            await asyncio.sleep(0.01 * complexity)
            return {
                "task_id": task_id,
                "complexity": complexity,
                "result": sum(range(complexity * 100)),
                "computed_at": time.time()
            }
        
        # 并发度测试
        concurrency_levels = [1, 5, 10, 20]
        throughput_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n测试并发度: {concurrency}")
            
            # 准备任务
            tasks = []
            start_time = time.time()
            
            # 创建并发任务
            for i in range(50):  # 每个并发级别执行50个任务
                task = compute_intensive_task(
                    performance_context,
                    task_id=i % 10,  # 重复任务ID以触发缓存命中
                    complexity=3
                )
                tasks.append(task)
                
                # 控制并发度
                if len(tasks) >= concurrency or i == 49:
                    # 执行当前批次的任务
                    await asyncio.gather(*tasks)
                    tasks = []
            
            total_time = time.time() - start_time
            throughput = 50 / total_time  # 任务/秒
            throughput_results[concurrency] = throughput
            
            print(f"  吞吐量: {throughput:.2f} tasks/sec")
            print(f"  总耗时: {total_time:.2f} seconds")
        
        # 验证并发性能不会严重下降
        baseline_throughput = throughput_results[1]
        for concurrency, throughput in throughput_results.items():
            if concurrency > 1:
                efficiency = throughput / (baseline_throughput * concurrency)
                print(f"并发度{concurrency}的效率: {efficiency:.2%}")
                # 效率不应该低于50%
                assert efficiency > 0.5, f"并发度{concurrency}时效率过低: {efficiency:.2%}"
    
    @pytest.mark.asyncio
    async def test_cache_memory_usage(self, performance_cache_setup):
        """缓存内存使用测试"""
        cache, monitor = performance_cache_setup
        
        # 创建不同大小的测试数据
        data_sizes = {
            "small": "x" * 100,        # 100 bytes
            "medium": "x" * 10000,     # 10 KB
            "large": "x" * 1000000,    # 1 MB
        }
        
        memory_usage = {}
        
        for size_name, data in data_sizes.items():
            # 清空缓存
            await cache.clear()
            
            # 设置多个相同大小的缓存项
            items_count = 100 if size_name != "large" else 10
            
            start_time = time.time()
            for i in range(items_count):
                await cache.set(f"{size_name}_{i}", {"data": data, "id": i})
            
            set_time = time.time() - start_time
            
            # 获取内存使用情况
            stats = await cache.get_stats()
            
            memory_usage[size_name] = {
                "items_count": items_count,
                "set_time": set_time,
                "avg_set_time": set_time / items_count * 1000,  # ms per item
                "cache_entries": stats.get("cache_entries", 0)
            }
            
            print(f"\n{size_name.title()}数据缓存测试:")
            print(f"  项目数量: {items_count}")
            print(f"  总设置时间: {set_time:.3f}s")
            print(f"  平均设置时间: {set_time / items_count * 1000:.2f}ms/项")
            print(f"  缓存条目数: {stats.get('cache_entries', 0)}")
        
        # 验证内存使用合理性
        assert memory_usage["small"]["avg_set_time"] < 10.0  # 小数据应该很快
        assert memory_usage["medium"]["avg_set_time"] < 50.0  # 中等数据应该适中
        assert memory_usage["large"]["avg_set_time"] < 200.0  # 大数据允许较慢
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_optimization(self, performance_cache_setup, performance_context):
        """缓存命中率优化测试"""
        cache, monitor = performance_cache_setup
        
        @cached_node(name="hit_rate_test", cache=cache)
        async def variable_computation(context: AgentContext, param: str, variant: int):
            await asyncio.sleep(0.02)  # 模拟计算时间
            return {
                "param": param,
                "variant": variant,
                "result": f"computed_{param}_{variant}",
                "timestamp": time.time()
            }
        
        # 模拟不同的访问模式
        access_patterns = {
            "sequential": [(f"param_{i}", i % 5) for i in range(100)],  # 顺序访问，高重复
            "random": [(f"param_{i % 20}", i % 10) for i in range(100)],  # 随机访问，中等重复
            "unique": [(f"param_{i}", i) for i in range(100)]  # 唯一访问，无重复
        }
        
        pattern_results = {}
        
        for pattern_name, params in access_patterns.items():
            # 清空缓存统计
            cache.stats.hit_count = 0
            cache.stats.miss_count = 0
            cache.stats.set_count = 0
            
            start_time = time.time()
            
            # 执行访问模式
            for param, variant in params:
                await variable_computation(performance_context, param, variant)
            
            total_time = time.time() - start_time
            
            # 获取统计信息
            stats = cache.stats.to_dict()
            
            pattern_results[pattern_name] = {
                "total_time": total_time,
                "avg_time_per_request": total_time / len(params) * 1000,  # ms
                "hit_rate": stats["hit_rate"],
                "hit_count": stats["hit_count"],
                "miss_count": stats["miss_count"],
                "set_count": stats["set_count"]
            }
            
            print(f"\n{pattern_name.title()}访问模式结果:")
            print(f"  总时间: {total_time:.3f}s")
            print(f"  平均请求时间: {total_time / len(params) * 1000:.2f}ms")
            print(f"  缓存命中率: {stats['hit_rate']:.2%}")
            print(f"  命中次数: {stats['hit_count']}")
            print(f"  未命中次数: {stats['miss_count']}")
        
        # 验证不同模式的预期性能特征
        assert pattern_results["sequential"]["hit_rate"] > 0.8  # 顺序访问应该有高命中率
        assert pattern_results["random"]["hit_rate"] > 0.5    # 随机访问应该有中等命中率
        assert pattern_results["unique"]["hit_rate"] == 0     # 唯一访问应该无命中
        
        # 验证缓存带来的性能提升
        sequential_time = pattern_results["sequential"]["avg_time_per_request"]
        unique_time = pattern_results["unique"]["avg_time_per_request"]
        
        # 高命中率的情况下应该显著更快
        assert sequential_time < unique_time * 0.7
    
    @pytest.mark.asyncio
    async def test_cache_scaling_limits(self, performance_context):
        """缓存扩展性限制测试"""
        # 测试不同最大条目数的性能表现
        max_entries_configs = [100, 1000, 10000]
        scaling_results = {}
        
        for max_entries in max_entries_configs:
            config = CacheConfig(
                backend="memory",
                max_entries=max_entries,
                ttl_default=3600
            )
            cache = create_node_cache(config)
            
            # 填充缓存到接近极限
            fill_count = int(max_entries * 0.9)
            
            # 测试填充时间
            start_time = time.time()
            for i in range(fill_count):
                await cache.set(f"scale_test_{i}", {"id": i, "data": f"data_{i}"})
            
            fill_time = time.time() - start_time
            
            # 测试访问性能
            access_times = []
            for i in range(min(100, fill_count)):
                start_time = time.time()
                await cache.get(f"scale_test_{i}")
                access_time = (time.time() - start_time) * 1000
                access_times.append(access_time)
            
            # 测试LRU淘汰性能
            eviction_start = time.time()
            for i in range(fill_count, fill_count + 50):
                await cache.set(f"scale_test_{i}", {"id": i, "data": f"data_{i}"})
            eviction_time = time.time() - eviction_start
            
            scaling_results[max_entries] = {
                "fill_time": fill_time,
                "fill_rate": fill_count / fill_time,
                "avg_access_time": statistics.mean(access_times),
                "eviction_time": eviction_time / 50 * 1000,  # ms per eviction
                "final_entries": (await cache.get_stats()).get("cache_entries", 0)
            }
            
            print(f"\n缓存规模测试 (max_entries={max_entries}):")
            print(f"  填充时间: {fill_time:.3f}s")
            print(f"  填充速率: {fill_count / fill_time:.1f} items/sec")
            print(f"  平均访问时间: {statistics.mean(access_times):.3f}ms")
            print(f"  平均淘汰时间: {eviction_time / 50 * 1000:.3f}ms/item")
            print(f"  最终条目数: {scaling_results[max_entries]['final_entries']}")
            
            await cache.clear()
        
        # 验证扩展性特征
        small_config = scaling_results[100]
        large_config = scaling_results[10000]
        
        # 填充速率不应该随规模显著下降
        fill_rate_ratio = large_config["fill_rate"] / small_config["fill_rate"]
        assert fill_rate_ratio > 0.5, f"大规模缓存填充速率下降过多: {fill_rate_ratio:.2%}"
        
        # 访问时间不应该随规模显著增加
        access_time_ratio = large_config["avg_access_time"] / small_config["avg_access_time"]
        assert access_time_ratio < 2.0, f"大规模缓存访问时间增长过多: {access_time_ratio:.2f}x"


@pytest.mark.asyncio
async def test_cache_stress_test():
    """缓存压力测试"""
    config = CacheConfig(
        backend="memory",
        max_entries=1000,
        ttl_default=3600
    )
    cache = create_node_cache(config)
    monitor = CacheMonitor(cache)
    
    # 启动监控
    await monitor.start_monitoring()
    
    try:
        context = AgentContext(
            user_id="stress_user",
            session_id="stress_session"
        )
        
        @cached_node(name="stress_test", cache=cache)
        async def stress_computation(context: AgentContext, task_id: int, group: str):
            # 模拟不同复杂度的计算
            complexity = hash(f"{task_id}_{group}") % 5 + 1
            await asyncio.sleep(0.001 * complexity)
            return {
                "task_id": task_id,
                "group": group,
                "complexity": complexity,
                "result": sum(range(complexity * 10))
            }
        
        # 高并发压力测试
        print("\n执行高并发压力测试...")
        start_time = time.time()
        
        # 创建大量并发任务
        tasks = []
        for i in range(500):
            task = stress_computation(
                context,
                task_id=i % 50,  # 重复任务以测试缓存效果
                group=f"group_{i % 10}"  # 分组以增加变化
            )
            tasks.append(task)
        
        # 以批次执行以控制并发度
        batch_size = 25
        completed_tasks = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)
            completed_tasks += len(batch)
            
            if completed_tasks % 100 == 0:
                print(f"  已完成 {completed_tasks} 个任务")
        
        total_time = time.time() - start_time
        throughput = len(tasks) / total_time
        
        # 获取最终统计
        final_stats = await monitor.get_detailed_stats()
        
        print(f"\n压力测试结果:")
        print(f"  总任务数: {len(tasks)}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  吞吐量: {throughput:.1f} tasks/sec")
        print(f"  缓存命中率: {final_stats['hit_rate']:.2%}")
        print(f"  缓存命中次数: {final_stats['hit_count']}")
        print(f"  平均获取延迟: {final_stats['avg_get_latency_ms']:.2f}ms")
        print(f"  平均设置延迟: {final_stats['avg_set_latency_ms']:.2f}ms")
        print(f"  错误次数: {final_stats['error_count']}")
        
        # 性能断言
        assert throughput > 50, f"吞吐量过低: {throughput:.1f} tasks/sec"
        assert final_stats["hit_rate"] > 0.5, f"缓存命中率过低: {final_stats['hit_rate']:.2%}"
        assert final_stats["error_count"] == 0, f"存在错误: {final_stats['error_count']}"
        assert final_stats["avg_get_latency_ms"] < 10, f"获取延迟过高: {final_stats['avg_get_latency_ms']:.2f}ms"
        
    finally:
        await monitor.stop_monitoring()
        await cache.clear()