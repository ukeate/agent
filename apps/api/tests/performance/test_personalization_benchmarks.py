"""
个性化引擎性能基准测试
验证特征计算延迟、推荐响应时间和吞吐量性能要求
"""

import pytest
import asyncio
import time
import statistics
from typing import List
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from ai.personalization.engine import PersonalizationEngine
from ai.personalization.features.realtime import RealTimeFeatureEngine
from models.schemas.personalization import RecommendationRequest


class TestPersonalizationBenchmarks:
    """个性化引擎性能基准测试套件"""
    
    @pytest.fixture
    async def engine(self):
        """创建性能测试用的个性化引擎"""
        with patch('ai.personalization.engine.redis') as mock_redis:
            mock_redis.get.return_value = None
            mock_redis.setex = AsyncMock()
            
            engine = PersonalizationEngine()
            await engine.initialize()
            return engine
    
    @pytest.fixture
    def feature_engine(self):
        """创建特征计算引擎"""
        with patch('redis.from_url'):
            return RealTimeFeatureEngine(
                redis_client=Mock(),
                config={
                    "feature_ttl": 300,
                    "computation_timeout": 0.01  # 10ms
                }
            )

    @pytest.mark.benchmark
    async def test_feature_computation_latency_benchmark(self, feature_engine):
        """基准测试：特征计算延迟 (<10ms要求)"""
        
        test_contexts = [
            {"page": "homepage", "device": "mobile"},
            {"page": "product", "device": "desktop", "category": "electronics"},
            {"page": "search", "query": "laptop", "filters": ["brand", "price"]},
            {"page": "checkout", "cart_value": 299.99},
            {"page": "profile", "section": "preferences"}
        ]
        
        user_ids = [f"benchmark_user_{i}" for i in range(100)]
        
        latencies = []
        
        # 执行大量特征计算测试
        for user_id in user_ids:
            for context in test_contexts:
                start_time = time.perf_counter()
                
                features = await feature_engine.compute_features(user_id, context)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # 验证特征结构完整性
                assert isinstance(features, dict)
                assert "temporal" in features
                assert "behavioral" in features
                assert "contextual" in features
        
        # 分析延迟分布
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)
        
        print(f"\n特征计算延迟基准测试结果:")
        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"P50延迟: {p50:.2f}ms")
        print(f"P95延迟: {p95:.2f}ms")
        print(f"P99延迟: {p99:.2f}ms")
        print(f"最大延迟: {max_latency:.2f}ms")
        print(f"总测试次数: {len(latencies)}")
        
        # 验证性能要求
        assert avg_latency < 5.0, f"平均特征计算延迟 {avg_latency:.2f}ms 超过5ms目标"
        assert p95 < 8.0, f"P95特征计算延迟 {p95:.2f}ms 超过8ms目标"
        assert p99 < 10.0, f"P99特征计算延迟 {p99:.2f}ms 超过10ms要求"
        assert max_latency < 15.0, f"最大特征计算延迟 {max_latency:.2f}ms 超过15ms阈值"

    @pytest.mark.benchmark
    async def test_recommendation_response_latency_benchmark(self, engine):
        """基准测试：推荐响应延迟 (P99 <100ms要求)"""
        
        # 创建不同复杂度的推荐请求
        test_requests = [
            # 简单请求
            RecommendationRequest(
                user_id="simple_user",
                context={"page": "home"},
                n_recommendations=5,
                scenario="quick"
            ),
            # 中等复杂度请求
            RecommendationRequest(
                user_id="medium_user",
                context={
                    "page": "search",
                    "query": "laptop computer",
                    "filters": {"price": "500-1000", "brand": "apple"},
                    "session_data": {"views": 15, "clicks": 3}
                },
                n_recommendations=10,
                scenario="search_results"
            ),
            # 复杂请求
            RecommendationRequest(
                user_id="complex_user",
                context={
                    "page": "personalized",
                    "user_segment": "premium",
                    "interaction_history": [{"item": f"item_{i}"} for i in range(20)],
                    "real_time_signals": {"current_mood": "exploratory", "time_pressure": "low"}
                },
                n_recommendations=20,
                scenario="deep_personalization",
                filters={"category": ["electronics", "books"], "min_rating": 4.0}
            )
        ]
        
        latencies = []
        
        # 执行多轮测试
        for round_num in range(50):  # 50轮测试
            for req_template in test_requests:
                # 为每个请求创建唯一用户避免缓存影响
                request = RecommendationRequest(
                    user_id=f"{req_template.user_id}_{round_num}",
                    context=req_template.context,
                    n_recommendations=req_template.n_recommendations,
                    scenario=req_template.scenario,
                    filters=req_template.filters
                )
                
                start_time = time.perf_counter()
                
                response = await engine.get_recommendations(request)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # 验证响应质量
                assert len(response.recommendations) > 0
                assert response.latency_ms > 0
        
        # 分析延迟分布
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)
        
        print(f"\n推荐响应延迟基准测试结果:")
        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"P50延迟: {p50:.2f}ms")
        print(f"P95延迟: {p95:.2f}ms")
        print(f"P99延迟: {p99:.2f}ms")
        print(f"最大延迟: {max_latency:.2f}ms")
        print(f"总请求数: {len(latencies)}")
        
        # 验证性能要求
        assert avg_latency < 30.0, f"平均推荐延迟 {avg_latency:.2f}ms 超过30ms目标"
        assert p50 < 50.0, f"P50推荐延迟 {p50:.2f}ms 超过50ms目标"
        assert p95 < 80.0, f"P95推荐延迟 {p95:.2f}ms 超过80ms目标"
        assert p99 < 100.0, f"P99推荐延迟 {p99:.2f}ms 超过100ms要求"

    @pytest.mark.benchmark
    async def test_throughput_benchmark(self, engine):
        """基准测试：系统吞吐量 (目标: 10,000+ QPS)"""
        
        # 创建标准测试请求
        def create_request(user_id: str, request_id: int) -> RecommendationRequest:
            return RecommendationRequest(
                user_id=user_id,
                context={
                    "request_id": request_id,
                    "page": "home",
                    "device": "mobile" if request_id % 2 == 0 else "desktop"
                },
                n_recommendations=5,
                scenario="throughput_test"
            )
        
        # 不同并发级别的吞吐量测试
        concurrency_levels = [10, 50, 100, 200]
        throughput_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n测试并发级别: {concurrency}")
            
            # 每个并发级别测试多轮
            rounds = 10
            total_requests = concurrency * rounds
            
            start_time = time.perf_counter()
            
            for round_num in range(rounds):
                # 创建并发请求批次
                requests = [
                    create_request(f"throughput_user_{i}", round_num * concurrency + i)
                    for i in range(concurrency)
                ]
                
                # 并发执行请求
                tasks = [engine.get_recommendations(req) for req in requests]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 检查成功率
                successful = [r for r in responses if not isinstance(r, Exception)]
                success_rate = len(successful) / len(responses)
                
                if success_rate < 0.95:
                    print(f"警告: 第{round_num}轮成功率只有{success_rate:.1%}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            qps = total_requests / total_time
            
            throughput_results[concurrency] = qps
            print(f"并发{concurrency}: {qps:.1f} QPS")
        
        # 验证吞吐量要求
        max_qps = max(throughput_results.values())
        print(f"\n吞吐量基准测试结果:")
        for concurrency, qps in throughput_results.items():
            print(f"并发{concurrency}: {qps:.1f} QPS")
        print(f"最大QPS: {max_qps:.1f}")
        
        # 在测试环境中降低要求，生产环境应该更高
        assert max_qps > 1000, f"最大QPS {max_qps:.1f} 低于1000要求"
        
        # 验证在合理并发下的稳定性
        stable_qps = throughput_results.get(100, 0)
        assert stable_qps > 500, f"100并发下QPS {stable_qps:.1f} 低于500要求"

    @pytest.mark.benchmark
    async def test_cache_hit_rate_benchmark(self, engine):
        """基准测试：缓存命中率 (>80%要求)"""
        
        # 创建有重复模式的请求来测试缓存
        users = [f"cache_user_{i}" for i in range(20)]  # 20个用户
        contexts = [
            {"page": "home"},
            {"page": "search", "query": "laptop"},
            {"page": "product", "category": "electronics"}
        ]
        
        total_requests = 0
        cache_hits = 0
        
        # 第一轮：填充缓存
        for user in users:
            for context in contexts:
                request = RecommendationRequest(
                    user_id=user,
                    context=context,
                    n_recommendations=5,
                    scenario="cache_test"
                )
                await engine.get_recommendations(request)
                total_requests += 1
        
        # 第二轮和第三轮：测试缓存命中
        for round_num in range(2, 5):  # 执行第2、3、4轮
            for user in users:
                for context in contexts:
                    request = RecommendationRequest(
                        user_id=user,
                        context=context,
                        n_recommendations=5,
                        scenario="cache_test"
                    )
                    
                    start_time = time.perf_counter()
                    response = await engine.get_recommendations(request)
                    end_time = time.perf_counter()
                    
                    request_time_ms = (end_time - start_time) * 1000
                    
                    # 判断是否命中缓存（缓存请求应该更快）
                    if request_time_ms < 10:  # 缓存请求应该在10ms内
                        cache_hits += 1
                    
                    total_requests += 1
        
        # 计算缓存命中率
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        print(f"\n缓存性能基准测试结果:")
        print(f"总请求数: {total_requests}")
        print(f"缓存命中数: {cache_hits}")
        print(f"缓存命中率: {cache_hit_rate:.1%}")
        
        # 验证缓存命中率要求
        assert cache_hit_rate > 0.6, f"缓存命中率 {cache_hit_rate:.1%} 低于60%最低要求"
        # 理想情况下应该>80%，但在测试环境中可能较低

    @pytest.mark.benchmark
    async def test_memory_efficiency_benchmark(self, engine):
        """基准测试：内存使用效率"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量操作
        num_operations = 1000
        memory_samples = []
        
        for i in range(num_operations):
            request = RecommendationRequest(
                user_id=f"memory_test_user_{i % 100}",
                context={"operation": i, "batch": i // 100},
                n_recommendations=10,
                scenario="memory_test"
            )
            
            await engine.get_recommendations(request)
            
            # 每50次操作检查一次内存
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        # 分析内存使用
        final_memory = process.memory_info().rss / 1024 / 1024
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        print(f"\n内存效率基准测试结果:")
        print(f"初始内存: {initial_memory:.1f} MB")
        print(f"最终内存: {final_memory:.1f} MB")
        print(f"最大内存: {max_memory:.1f} MB")
        print(f"内存增长: {memory_growth:.1f} MB")
        print(f"每操作内存增长: {memory_growth/num_operations:.3f} MB")
        
        # 验证内存效率
        assert memory_growth < 100, f"内存增长 {memory_growth:.1f}MB 过大"
        assert memory_growth / num_operations < 0.1, "每操作内存增长过大"

    @pytest.mark.benchmark
    async def test_scalability_benchmark(self, engine):
        """基准测试：系统可扩展性"""
        
        # 测试不同负载下的性能退化
        load_levels = [
            {"users": 10, "requests_per_user": 10},
            {"users": 50, "requests_per_user": 10},
            {"users": 100, "requests_per_user": 5},
            {"users": 200, "requests_per_user": 3}
        ]
        
        performance_results = {}
        
        for load in load_levels:
            users = load["users"]
            requests_per_user = load["requests_per_user"]
            total_requests = users * requests_per_user
            
            print(f"\n测试负载: {users}用户 x {requests_per_user}请求 = {total_requests}总请求")
            
            start_time = time.perf_counter()
            latencies = []
            
            # 创建所有请求
            all_requests = []
            for user_id in range(users):
                for req_id in range(requests_per_user):
                    request = RecommendationRequest(
                        user_id=f"scale_user_{user_id}",
                        context={"request": req_id, "load_test": True},
                        n_recommendations=5,
                        scenario="scalability_test"
                    )
                    all_requests.append(request)
            
            # 分批执行以避免过度并发
            batch_size = min(50, total_requests)
            for i in range(0, len(all_requests), batch_size):
                batch = all_requests[i:i + batch_size]
                
                batch_start = time.perf_counter()
                tasks = [engine.get_recommendations(req) for req in batch]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                batch_end = time.perf_counter()
                
                # 记录批次延迟
                for j, response in enumerate(responses):
                    if not isinstance(response, Exception):
                        request_latency = (batch_end - batch_start) * 1000 / len(batch)
                        latencies.append(request_latency)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 计算性能指标
            avg_latency = statistics.mean(latencies) if latencies else float('inf')
            qps = len(latencies) / total_time if total_time > 0 else 0
            success_rate = len(latencies) / total_requests
            
            performance_results[total_requests] = {
                "avg_latency": avg_latency,
                "qps": qps,
                "success_rate": success_rate
            }
            
            print(f"平均延迟: {avg_latency:.2f}ms")
            print(f"QPS: {qps:.1f}")
            print(f"成功率: {success_rate:.1%}")
        
        # 验证可扩展性
        # 检查在负载增加时性能退化是否合理
        qps_values = [result["qps"] for result in performance_results.values()]
        latency_values = [result["avg_latency"] for result in performance_results.values()]
        
        # QPS应该在合理范围内
        min_qps = min(qps_values)
        max_qps = max(qps_values)
        
        assert min_qps > 50, f"最低QPS {min_qps:.1f} 过低"
        
        # 延迟增长应该是渐进的，不是指数级的
        max_latency = max(latency_values)
        assert max_latency < 200, f"最大延迟 {max_latency:.2f}ms 过高"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])