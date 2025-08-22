"""
强化学习系统性能测试套件

测试系统在高负载下的性能表现，包括：
- 响应时间基准测试
- 吞吐量测试
- 内存使用测试
- 并发性能测试
- 压力测试
"""

import pytest
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median

from ai.reinforcement_learning.recommendation_engine import (
    BanditRecommendationEngine,
    AlgorithmType,
    RecommendationRequest,
    FeedbackData
)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        self.response_times = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.response_times = []
        
    def record_metrics(self):
        """记录指标"""
        process = psutil.Process()
        self.cpu_usage.append(process.cpu_percent())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        
    def stop_monitoring(self):
        """停止监控"""
        self.end_time = time.time()
        
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "avg_cpu_percent": mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_mb": mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "total_responses": len(self.response_times),
            "avg_response_time_ms": mean(self.response_times) if self.response_times else 0,
            "p50_response_time_ms": median(self.response_times) if self.response_times else 0,
            "p95_response_time_ms": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0,
            "p99_response_time_ms": sorted(self.response_times)[int(len(self.response_times) * 0.99)] if self.response_times else 0,
        }


class TestRLSystemPerformance:
    """强化学习系统性能测试"""
    
    @pytest.fixture
    async def high_performance_engine(self):
        """创建高性能配置的推荐引擎"""
        engine = BanditRecommendationEngine(
            default_algorithm=AlgorithmType.UCB,
            enable_cold_start=True,
            enable_evaluation=True,
            cache_ttl_seconds=600,  # 更长的缓存时间
            max_cache_size=50000   # 更大的缓存
        )
        
        await engine.initialize_algorithms(n_items=1000)
        return engine
    
    @pytest.fixture
    def performance_monitor(self):
        """创建性能监控器"""
        return PerformanceMonitor()
    
    @pytest.mark.performance
    async def test_response_time_benchmarks(self, high_performance_engine, performance_monitor):
        """响应时间基准测试"""
        
        performance_monitor.start_monitoring()
        
        # 测试不同场景下的响应时间
        test_scenarios = [
            {"name": "cold_start", "user_id": "new_user", "context": {"new": True}},
            {"name": "cached_request", "user_id": "user_1", "context": {"cached": True}},
            {"name": "contextual", "user_id": "user_2", "context": {"age": 25, "location": "beijing", "interests": ["tech", "sports"]}},
            {"name": "simple", "user_id": "user_3", "context": None},
        ]
        
        scenario_results = {}
        
        for scenario in test_scenarios:
            response_times = []
            
            # 每个场景测试100次
            for i in range(100):
                request = RecommendationRequest(
                    user_id=f"{scenario['user_id']}_{i}",
                    context=scenario["context"],
                    num_recommendations=10
                )
                
                start_time = time.time()
                response = await high_performance_engine.get_recommendations(request)
                response_time = (time.time() - start_time) * 1000
                
                response_times.append(response_time)
                performance_monitor.response_times.append(response_time)
                performance_monitor.record_metrics()
                
                # 验证响应质量
                assert len(response.recommendations) == 10
                assert response.processing_time_ms > 0
            
            # 计算场景统计
            scenario_results[scenario["name"]] = {
                "avg": mean(response_times),
                "p50": median(response_times),
                "p95": sorted(response_times)[95],
                "p99": sorted(response_times)[99],
                "max": max(response_times),
                "min": min(response_times)
            }
        
        performance_monitor.stop_monitoring()
        
        # 验证性能要求
        for scenario_name, stats in scenario_results.items():
            print(f"\n{scenario_name} 性能统计:")
            print(f"  平均响应时间: {stats['avg']:.2f}ms")
            print(f"  P50: {stats['p50']:.2f}ms")
            print(f"  P95: {stats['p95']:.2f}ms")
            print(f"  P99: {stats['p99']:.2f}ms")
            
            # 性能断言
            assert stats["p50"] < 50, f"{scenario_name} P50 响应时间超标: {stats['p50']:.2f}ms"
            assert stats["p95"] < 100, f"{scenario_name} P95 响应时间超标: {stats['p95']:.2f}ms"
            assert stats["p99"] < 200, f"{scenario_name} P99 响应时间超标: {stats['p99']:.2f}ms"
        
        # 打印整体性能摘要
        summary = performance_monitor.get_summary()
        print(f"\n整体性能摘要:")
        print(f"  总响应数: {summary['total_responses']}")
        print(f"  平均CPU使用率: {summary['avg_cpu_percent']:.2f}%")
        print(f"  平均内存使用: {summary['avg_memory_mb']:.2f}MB")
    
    @pytest.mark.performance
    async def test_throughput_benchmarks(self, high_performance_engine, performance_monitor):
        """吞吐量基准测试"""
        
        performance_monitor.start_monitoring()
        
        # 测试不同并发级别的吞吐量
        concurrency_levels = [1, 5, 10, 20, 50]
        throughput_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n测试并发级别: {concurrency}")
            
            async def make_request(request_id: int):
                """单个请求"""
                request = RecommendationRequest(
                    user_id=f"user_{request_id}",
                    context={"concurrency_test": True, "level": concurrency},
                    num_recommendations=5
                )
                
                start_time = time.time()
                response = await high_performance_engine.get_recommendations(request)
                response_time = (time.time() - start_time) * 1000
                
                performance_monitor.response_times.append(response_time)
                return response
            
            # 创建并发任务
            num_requests_per_level = 100
            start_time = time.time()
            
            # 分批执行以控制并发
            batch_size = concurrency
            total_requests = num_requests_per_level
            
            for batch_start in range(0, total_requests, batch_size):
                batch_end = min(batch_start + batch_size, total_requests)
                tasks = [
                    make_request(batch_start + i) 
                    for i in range(batch_end - batch_start)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                # 验证响应
                for response in responses:
                    assert len(response.recommendations) == 5
                
                performance_monitor.record_metrics()
            
            total_time = time.time() - start_time
            throughput = num_requests_per_level / total_time
            
            throughput_results[concurrency] = {
                "requests_per_second": throughput,
                "total_time": total_time,
                "avg_response_time": mean(performance_monitor.response_times[-num_requests_per_level:])
            }
            
            print(f"  吞吐量: {throughput:.2f} req/s")
            print(f"  总耗时: {total_time:.2f}s")
        
        performance_monitor.stop_monitoring()
        
        # 验证吞吐量要求
        max_throughput = max(result["requests_per_second"] for result in throughput_results.values())
        print(f"\n最大吞吐量: {max_throughput:.2f} req/s")
        
        # 要求支持至少1000 QPS（在合理并发下）
        assert max_throughput > 100, f"吞吐量不足: {max_throughput:.2f} req/s"
    
    @pytest.mark.performance  
    async def test_memory_usage_under_load(self, high_performance_engine, performance_monitor):
        """负载下内存使用测试"""
        
        performance_monitor.start_monitoring()
        
        # 记录初始内存使用
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 执行大量推荐请求
        num_users = 1000
        requests_per_user = 10
        
        for user_id in range(num_users):
            for request_id in range(requests_per_user):
                request = RecommendationRequest(
                    user_id=f"memory_test_user_{user_id}",
                    context={
                        "request_id": request_id,
                        "data": [i for i in range(100)]  # 添加一些数据
                    },
                    num_recommendations=5
                )
                
                response = await high_performance_engine.get_recommendations(request)
                assert len(response.recommendations) == 5
                
                # 定期记录内存使用
                if (user_id * requests_per_user + request_id) % 100 == 0:
                    performance_monitor.record_metrics()
                    
                    # 定期强制垃圾回收
                    if (user_id * requests_per_user + request_id) % 500 == 0:
                        gc.collect()
        
        performance_monitor.stop_monitoring()
        
        # 最终内存使用
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        summary = performance_monitor.get_summary()
        
        print(f"\n内存使用统计:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  最终内存: {final_memory:.2f}MB")
        print(f"  内存增长: {memory_growth:.2f}MB")
        print(f"  最大内存: {summary['max_memory_mb']:.2f}MB")
        print(f"  平均内存: {summary['avg_memory_mb']:.2f}MB")
        
        # 验证内存要求（单实例 < 4GB）
        assert summary["max_memory_mb"] < 4096, f"内存使用超标: {summary['max_memory_mb']:.2f}MB"
        
        # 验证内存增长合理（不应该有严重内存泄漏）
        assert memory_growth < 1000, f"内存增长过大: {memory_growth:.2f}MB"
    
    @pytest.mark.performance
    async def test_stress_testing(self, high_performance_engine, performance_monitor):
        """压力测试"""
        
        performance_monitor.start_monitoring()
        
        # 高强度负载参数
        duration_seconds = 30  # 测试持续时间
        target_qps = 50       # 目标QPS
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        async def stress_worker():
            """压力测试工作进程"""
            nonlocal request_count, error_count
            
            while time.time() - start_time < duration_seconds:
                try:
                    request = RecommendationRequest(
                        user_id=f"stress_user_{request_count % 100}",
                        context={"stress_test": True, "timestamp": time.time()},
                        num_recommendations=5
                    )
                    
                    req_start = time.time()
                    response = await high_performance_engine.get_recommendations(request)
                    req_time = (time.time() - req_start) * 1000
                    
                    performance_monitor.response_times.append(req_time)
                    request_count += 1
                    
                    # 验证响应
                    assert len(response.recommendations) == 5
                    
                    # 控制请求速率
                    await asyncio.sleep(1.0 / target_qps)
                    
                except Exception as e:
                    error_count += 1
                    print(f"请求失败: {e}")
                
                # 定期记录指标
                if request_count % 10 == 0:
                    performance_monitor.record_metrics()
        
        # 启动多个工作进程
        num_workers = 5
        tasks = [stress_worker() for _ in range(num_workers)]
        
        await asyncio.gather(*tasks)
        
        performance_monitor.stop_monitoring()
        
        # 计算性能指标
        actual_duration = time.time() - start_time
        actual_qps = request_count / actual_duration
        error_rate = error_count / request_count if request_count > 0 else 0
        
        summary = performance_monitor.get_summary()
        
        print(f"\n压力测试结果:")
        print(f"  测试时长: {actual_duration:.2f}s")
        print(f"  总请求数: {request_count}")
        print(f"  实际QPS: {actual_qps:.2f}")
        print(f"  错误数: {error_count}")
        print(f"  错误率: {error_rate:.2%}")
        print(f"  平均响应时间: {summary['avg_response_time_ms']:.2f}ms")
        print(f"  P95响应时间: {summary['p95_response_time_ms']:.2f}ms")
        print(f"  平均CPU: {summary['avg_cpu_percent']:.2f}%")
        print(f"  峰值CPU: {summary['max_cpu_percent']:.2f}%")
        print(f"  平均内存: {summary['avg_memory_mb']:.2f}MB")
        
        # 验证压力测试要求
        assert actual_qps >= target_qps * 0.8, f"QPS不足: {actual_qps:.2f} < {target_qps * 0.8:.2f}"
        assert error_rate < 0.01, f"错误率过高: {error_rate:.2%}"
        assert summary["avg_response_time_ms"] < 200, f"平均响应时间过长: {summary['avg_response_time_ms']:.2f}ms"
        assert summary["p95_response_time_ms"] < 500, f"P95响应时间过长: {summary['p95_response_time_ms']:.2f}ms"
        assert summary["max_cpu_percent"] < 90, f"CPU使用率过高: {summary['max_cpu_percent']:.2f}%"
    
    @pytest.mark.performance
    async def test_algorithm_performance_comparison(self, performance_monitor):
        """算法性能比较测试"""
        
        algorithms = [
            AlgorithmType.UCB,
            AlgorithmType.THOMPSON_SAMPLING,
            AlgorithmType.EPSILON_GREEDY,
            AlgorithmType.LINEAR_CONTEXTUAL
        ]
        
        algorithm_results = {}
        
        for algorithm in algorithms:
            print(f"\n测试算法: {algorithm.value}")
            
            # 为每个算法创建独立的引擎
            engine = BanditRecommendationEngine(
                default_algorithm=algorithm,
                enable_cold_start=False,  # 关闭冷启动以纯测试算法性能
                enable_evaluation=False,  # 关闭评估以减少开销
                cache_ttl_seconds=0       # 关闭缓存以测试真实算法性能
            )
            await engine.initialize_algorithms(n_items=100)
            
            performance_monitor.start_monitoring()
            response_times = []
            
            # 测试算法性能
            num_requests = 200
            for i in range(num_requests):
                request = RecommendationRequest(
                    user_id=f"algo_test_user_{i % 20}",
                    context={"feature_1": i % 10, "feature_2": (i * 2) % 5} if algorithm == AlgorithmType.LINEAR_CONTEXTUAL else None,
                    num_recommendations=5
                )
                
                start_time = time.time()
                response = await engine.get_recommendations(request)
                response_time = (time.time() - start_time) * 1000
                
                response_times.append(response_time)
                
                # 模拟反馈
                if i % 5 == 0:  # 每5个请求给一次反馈
                    feedback = FeedbackData(
                        user_id=request.user_id,
                        item_id=response.recommendations[0]["item_id"],
                        feedback_type="click",
                        feedback_value=0.6,
                        context=request.context,
                        timestamp=datetime.now()
                    )
                    await engine.process_feedback(feedback)
                
                performance_monitor.record_metrics()
            
            performance_monitor.stop_monitoring()
            
            # 记录算法性能
            algorithm_results[algorithm.value] = {
                "avg_response_time": mean(response_times),
                "p50_response_time": median(response_times),
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
                "max_response_time": max(response_times),
                "min_response_time": min(response_times)
            }
            
            print(f"  平均响应时间: {algorithm_results[algorithm.value]['avg_response_time']:.2f}ms")
            print(f"  P95响应时间: {algorithm_results[algorithm.value]['p95_response_time']:.2f}ms")
        
        # 性能比较分析
        print(f"\n算法性能比较:")
        for algo, stats in algorithm_results.items():
            print(f"  {algo}: 平均{stats['avg_response_time']:.2f}ms, P95:{stats['p95_response_time']:.2f}ms")
        
        # 验证所有算法性能都在可接受范围内
        for algo, stats in algorithm_results.items():
            assert stats["avg_response_time"] < 100, f"{algo}平均响应时间过长: {stats['avg_response_time']:.2f}ms"
            assert stats["p95_response_time"] < 200, f"{algo} P95响应时间过长: {stats['p95_response_time']:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])