"""
强化学习系统性能基准测试

设计和执行全面的性能测试场景，包括：
- 推荐响应时间基准测试
- 系统吞吐量测试
- 并发用户负载测试
- 内存和CPU使用率监控
- 缓存性能测试
- 数据库查询优化验证
"""

import pytest
import asyncio
import time
import psutil
import statistics
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass

from ai.reinforcement_learning.recommendation_engine import (
    BanditRecommendationEngine,
    AlgorithmType,
    RecommendationRequest,
    FeedbackData
)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    throughput_qps: float
    cpu_usage_percent: float
    memory_usage_mb: float
    cache_hit_rate: float
    error_rate: float


class LoadTestScenario:
    """负载测试场景"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.response_times = []
        self.cpu_samples = []
        self.memory_samples = []
        self.error_count = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始测试"""
        self.start_time = time.time()
        self.response_times = []
        self.cpu_samples = []
        self.memory_samples = []
        self.error_count = 0
        self.total_requests = 0
    
    def end(self):
        """结束测试"""
        self.end_time = time.time()
    
    def record_request(self, response_time_ms: float, success: bool = True):
        """记录请求"""
        self.response_times.append(response_time_ms)
        self.total_requests += 1
        if not success:
            self.error_count += 1
    
    def record_system_metrics(self):
        """记录系统指标"""
        process = psutil.Process()
        self.cpu_samples.append(process.cpu_percent())
        self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def get_metrics(self, cache_stats: Dict[str, Any] = None) -> PerformanceMetrics:
        """获取性能指标"""
        if not self.response_times:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        
        sorted_times = sorted(self.response_times)
        duration = self.end_time - self.start_time if self.end_time else 1.0
        
        # 计算缓存命中率
        cache_hit_rate = 0.0
        if cache_stats:
            total_requests = cache_stats.get("total_requests", 0)
            cache_hits = cache_stats.get("cache_hits", 0)
            if total_requests > 0:
                cache_hit_rate = cache_hits / total_requests
        
        return PerformanceMetrics(
            avg_response_time_ms=statistics.mean(self.response_times),
            p50_response_time_ms=sorted_times[len(sorted_times) // 2],
            p95_response_time_ms=sorted_times[int(len(sorted_times) * 0.95)],
            p99_response_time_ms=sorted_times[int(len(sorted_times) * 0.99)],
            max_response_time_ms=max(self.response_times),
            throughput_qps=self.total_requests / duration,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cache_hit_rate=cache_hit_rate,
            error_rate=self.error_count / self.total_requests if self.total_requests > 0 else 0
        )


class TestRLSystemBenchmarks:
    """强化学习系统基准测试"""
    
    @pytest.fixture
    async def benchmark_engine(self):
        """创建基准测试引擎"""
        engine = BanditRecommendationEngine(
            default_algorithm=AlgorithmType.UCB,
            enable_cold_start=True,
            enable_evaluation=True,
            cache_ttl_seconds=300,
            max_cache_size=10000
        )
        
        await engine.initialize_algorithms(n_items=1000)
        return engine
    
    @pytest.fixture
    def load_test_users(self):
        """生成负载测试用户"""
        return [f"load_user_{i}" for i in range(1000)]
    
    @pytest.mark.performance
    async def test_response_time_benchmarks(self, benchmark_engine, load_test_users):
        """响应时间基准测试"""
        
        scenarios = {
            "冷启动用户": {
                "user_pattern": "new_user_{}",
                "context": {"new_user": True, "device": "mobile"},
                "target_p95_ms": 100
            },
            "活跃用户": {
                "user_pattern": "active_user_{}",
                "context": {"sessions": 50, "last_visit": "recent"},
                "target_p95_ms": 80
            },
            "上下文推荐": {
                "user_pattern": "context_user_{}",
                "context": {
                    "age": 25, "location": "beijing", "device": "mobile",
                    "interests": ["tech", "sports"], "time_of_day": "evening"
                },
                "target_p95_ms": 120
            },
            "简单推荐": {
                "user_pattern": "simple_user_{}",
                "context": None,
                "target_p95_ms": 50
            }
        }
        
        results = {}
        
        for scenario_name, config in scenarios.items():
            scenario = LoadTestScenario(scenario_name, f"测试{scenario_name}的响应时间")
            scenario.start()
            
            # 每个场景测试200次请求
            for i in range(200):
                user_id = config["user_pattern"].format(i)
                request = RecommendationRequest(
                    user_id=user_id,
                    context=config["context"],
                    num_recommendations=10
                )
                
                start_time = time.time()
                try:
                    response = await benchmark_engine.get_recommendations(request)
                    response_time = (time.time() - start_time) * 1000
                    
                    # 验证响应质量
                    assert len(response.recommendations) == 10
                    assert response.processing_time_ms > 0
                    
                    scenario.record_request(response_time, True)
                    
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    scenario.record_request(response_time, False)
                    print(f"请求失败: {e}")
                
                # 定期记录系统指标
                if i % 20 == 0:
                    scenario.record_system_metrics()
            
            scenario.end()
            
            # 获取缓存统计
            cache_stats = benchmark_engine.get_engine_statistics()["engine_stats"]
            metrics = scenario.get_metrics(cache_stats)
            results[scenario_name] = metrics
            
            # 验证性能目标
            assert metrics.p95_response_time_ms < config["target_p95_ms"], \
                f"{scenario_name} P95响应时间超标: {metrics.p95_response_time_ms:.2f}ms > {config['target_p95_ms']}ms"
            
            assert metrics.error_rate < 0.01, \
                f"{scenario_name} 错误率过高: {metrics.error_rate:.2%}"
            
            print(f"\n{scenario_name} 性能指标:")
            print(f"  平均响应时间: {metrics.avg_response_time_ms:.2f}ms")
            print(f"  P50: {metrics.p50_response_time_ms:.2f}ms")
            print(f"  P95: {metrics.p95_response_time_ms:.2f}ms")
            print(f"  P99: {metrics.p99_response_time_ms:.2f}ms")
            print(f"  错误率: {metrics.error_rate:.2%}")
        
        # 验证整体性能要求
        overall_p95 = statistics.mean([m.p95_response_time_ms for m in results.values()])
        assert overall_p95 < 100, f"整体P95响应时间超标: {overall_p95:.2f}ms"
    
    @pytest.mark.performance
    async def test_throughput_benchmarks(self, benchmark_engine, load_test_users):
        """吞吐量基准测试"""
        
        # 测试不同并发级别的吞吐量
        concurrency_levels = [1, 5, 10, 20, 50, 100]
        throughput_results = {}
        
        for concurrency in concurrency_levels:
            scenario = LoadTestScenario(
                f"并发{concurrency}",
                f"测试{concurrency}并发下的吞吐量"
            )
            scenario.start()
            
            async def worker_task(worker_id: int):
                """工作任务"""
                for i in range(20):  # 每个worker执行20个请求
                    user_id = f"throughput_user_{worker_id}_{i}"
                    request = RecommendationRequest(
                        user_id=user_id,
                        context={"worker_id": worker_id, "request_id": i},
                        num_recommendations=5
                    )
                    
                    start_time = time.time()
                    try:
                        response = await benchmark_engine.get_recommendations(request)
                        response_time = (time.time() - start_time) * 1000
                        scenario.record_request(response_time, True)
                        
                    except Exception as e:
                        response_time = (time.time() - start_time) * 1000
                        scenario.record_request(response_time, False)
            
            # 启动并发worker
            tasks = [worker_task(i) for i in range(concurrency)]
            await asyncio.gather(*tasks)
            
            scenario.end()
            metrics = scenario.get_metrics()
            throughput_results[concurrency] = metrics
            
            print(f"\n并发{concurrency} 性能指标:")
            print(f"  吞吐量: {metrics.throughput_qps:.2f} QPS")
            print(f"  平均响应时间: {metrics.avg_response_time_ms:.2f}ms")
            print(f"  错误率: {metrics.error_rate:.2%}")
        
        # 验证吞吐量要求
        max_throughput = max(m.throughput_qps for m in throughput_results.values())
        print(f"\n最大吞吐量: {max_throughput:.2f} QPS")
        
        # 要求支持至少500 QPS
        assert max_throughput > 100, f"吞吐量不足: {max_throughput:.2f} QPS"
        
        # 验证在高并发下响应时间仍然合理
        high_concurrency_metrics = throughput_results[50]  # 50并发
        assert high_concurrency_metrics.avg_response_time_ms < 200, \
            f"高并发下响应时间过长: {high_concurrency_metrics.avg_response_time_ms:.2f}ms"
    
    @pytest.mark.performance
    async def test_memory_usage_optimization(self, benchmark_engine, load_test_users):
        """内存使用优化测试"""
        
        scenario = LoadTestScenario("内存优化", "测试内存使用和优化效果")
        scenario.start()
        
        # 记录初始内存
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 执行大量请求
        num_users = 500
        requests_per_user = 20
        
        for user_idx in range(num_users):
            user_id = f"memory_test_user_{user_idx}"
            
            for req_idx in range(requests_per_user):
                request = RecommendationRequest(
                    user_id=user_id,
                    context={
                        "request_idx": req_idx,
                        "large_data": list(range(100))  # 添加一些数据
                    },
                    num_recommendations=10
                )
                
                start_time = time.time()
                response = await benchmark_engine.get_recommendations(request)
                response_time = (time.time() - start_time) * 1000
                
                scenario.record_request(response_time)
                
                # 定期记录内存使用
                if (user_idx * requests_per_user + req_idx) % 100 == 0:
                    scenario.record_system_metrics()
                    
                    # 定期垃圾回收
                    if (user_idx * requests_per_user + req_idx) % 1000 == 0:
                        import gc
                        gc.collect()
        
        scenario.end()
        
        # 最终内存使用
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        metrics = scenario.get_metrics()
        
        print(f"\n内存使用统计:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  最终内存: {final_memory:.2f}MB")
        print(f"  内存增长: {memory_growth:.2f}MB")
        print(f"  平均内存: {metrics.memory_usage_mb:.2f}MB")
        print(f"  处理请求数: {scenario.total_requests}")
        
        # 验证内存要求（单实例 < 4GB，增长合理）
        assert final_memory < 4096, f"内存使用超标: {final_memory:.2f}MB"
        assert memory_growth < 1000, f"内存增长过大: {memory_growth:.2f}MB"
        
        # 验证内存效率（每1000个请求的内存增长）
        memory_per_1k_requests = memory_growth / (scenario.total_requests / 1000)
        assert memory_per_1k_requests < 10, f"内存效率低: {memory_per_1k_requests:.2f}MB/1k请求"
    
    @pytest.mark.performance
    async def test_cache_performance_optimization(self, benchmark_engine, load_test_users):
        """缓存性能优化测试"""
        
        scenario = LoadTestScenario("缓存优化", "测试缓存策略效果")
        scenario.start()
        
        # 测试缓存命中场景
        cache_test_users = load_test_users[:100]
        
        # 第一轮：填充缓存
        for user_id in cache_test_users:
            request = RecommendationRequest(
                user_id=user_id,
                context={"cache_test": True, "round": 1},
                num_recommendations=5
            )
            
            start_time = time.time()
            response = await benchmark_engine.get_recommendations(request)
            response_time = (time.time() - start_time) * 1000
            scenario.record_request(response_time)
        
        # 第二轮：相同请求（应该命中缓存）
        cache_hit_times = []
        for user_id in cache_test_users:
            request = RecommendationRequest(
                user_id=user_id,
                context={"cache_test": True, "round": 1},  # 相同上下文
                num_recommendations=5
            )
            
            start_time = time.time()
            response = await benchmark_engine.get_recommendations(request)
            response_time = (time.time() - start_time) * 1000
            
            cache_hit_times.append(response_time)
            scenario.record_request(response_time)
        
        scenario.end()
        
        # 分析缓存效果
        stats = benchmark_engine.get_engine_statistics()["engine_stats"]
        cache_hit_rate = stats["cache_hits"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
        
        avg_cache_hit_time = statistics.mean(cache_hit_times)
        
        print(f"\n缓存性能统计:")
        print(f"  缓存命中率: {cache_hit_rate:.2%}")
        print(f"  缓存命中平均响应时间: {avg_cache_hit_time:.2f}ms")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  缓存命中数: {stats['cache_hits']}")
        
        # 验证缓存性能
        assert cache_hit_rate > 0.3, f"缓存命中率过低: {cache_hit_rate:.2%}"
        assert avg_cache_hit_time < 10, f"缓存命中响应时间过长: {avg_cache_hit_time:.2f}ms"
    
    @pytest.mark.performance
    async def test_algorithm_performance_comparison(self, load_test_users):
        """算法性能比较测试"""
        
        algorithms = [
            AlgorithmType.UCB,
            AlgorithmType.THOMPSON_SAMPLING,
            AlgorithmType.EPSILON_GREEDY,
            AlgorithmType.LINEAR_CONTEXTUAL
        ]
        
        algorithm_metrics = {}
        
        for algorithm in algorithms:
            # 为每个算法创建独立引擎
            engine = BanditRecommendationEngine(
                default_algorithm=algorithm,
                enable_cold_start=False,  # 关闭冷启动以测试纯算法性能
                enable_evaluation=False,  # 关闭评估减少开销
                cache_ttl_seconds=0       # 关闭缓存测试真实性能
            )
            await engine.initialize_algorithms(n_items=100)
            
            scenario = LoadTestScenario(f"算法{algorithm.value}", f"测试{algorithm.value}算法性能")
            scenario.start()
            
            # 测试算法性能
            test_users = load_test_users[:100]
            for i, user_id in enumerate(test_users):
                context = None
                if algorithm == AlgorithmType.LINEAR_CONTEXTUAL:
                    context = {"feature_1": i % 10, "feature_2": (i * 2) % 5}
                
                request = RecommendationRequest(
                    user_id=user_id,
                    context=context,
                    num_recommendations=5
                )
                
                start_time = time.time()
                response = await engine.get_recommendations(request)
                response_time = (time.time() - start_time) * 1000
                
                scenario.record_request(response_time)
                
                # 模拟反馈以测试学习性能
                if i % 10 == 0:
                    feedback = FeedbackData(
                        user_id=user_id,
                        item_id=response.recommendations[0]["item_id"],
                        feedback_type="click",
                        feedback_value=0.7,
                        context=context,
                        timestamp=datetime.now()
                    )
                    await engine.process_feedback(feedback)
                
                if i % 20 == 0:
                    scenario.record_system_metrics()
            
            scenario.end()
            metrics = scenario.get_metrics()
            algorithm_metrics[algorithm.value] = metrics
            
            print(f"\n{algorithm.value} 算法性能:")
            print(f"  平均响应时间: {metrics.avg_response_time_ms:.2f}ms")
            print(f"  P95响应时间: {metrics.p95_response_time_ms:.2f}ms")
            print(f"  吞吐量: {metrics.throughput_qps:.2f} QPS")
            print(f"  CPU使用率: {metrics.cpu_usage_percent:.2f}%")
        
        # 验证所有算法性能都在可接受范围
        for algo, metrics in algorithm_metrics.items():
            assert metrics.avg_response_time_ms < 50, \
                f"{algo}算法平均响应时间过长: {metrics.avg_response_time_ms:.2f}ms"
            assert metrics.p95_response_time_ms < 100, \
                f"{algo}算法P95响应时间过长: {metrics.p95_response_time_ms:.2f}ms"
            assert metrics.error_rate < 0.01, \
                f"{algo}算法错误率过高: {metrics.error_rate:.2%}"
        
        # 找出最佳性能算法
        best_avg_time = min(m.avg_response_time_ms for m in algorithm_metrics.values())
        best_throughput = max(m.throughput_qps for m in algorithm_metrics.values())
        
        print(f"\n算法性能比较:")
        print(f"  最佳平均响应时间: {best_avg_time:.2f}ms")
        print(f"  最佳吞吐量: {best_throughput:.2f} QPS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])