"""
负载测试和性能基准测试
"""

import pytest
import asyncio
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import aiohttp
import numpy as np
from typing import List, Dict, Any
import json
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def record_request(self, response_time: float, success: bool):
        """记录请求"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        total_requests = len(self.response_times)
        
        return {
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0,
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "mean_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
            "p99_response_time": sorted_times[int(len(sorted_times) * 0.99)],
            "throughput": total_requests / (self.end_time - self.start_time) if self.end_time else 0
        }

class LoadTester:
    """负载测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics = PerformanceMetrics()
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, method: str = "GET", data: Dict = None):
        """发送单个请求"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    await response.text()
                    response_time = time.time() - start_time
                    self.metrics.record_request(response_time, response.status == 200)
                    return response.status
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    await response.text()
                    response_time = time.time() - start_time
                    self.metrics.record_request(response_time, response.status == 200)
                    return response.status
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(response_time, False)
            return None
    
    async def run_concurrent_requests(self, endpoint: str, num_requests: int, concurrency: int, method: str = "GET", data_generator=None):
        """运行并发请求"""
        self.metrics = PerformanceMetrics()
        self.metrics.start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                data = data_generator(i) if data_generator else None
                task = self.make_request(session, endpoint, method, data)
                tasks.append(task)
                
                # 控制并发数
                if len(tasks) >= concurrency:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            # 处理剩余的请求
            if tasks:
                await asyncio.gather(*tasks)
        
        self.metrics.end_time = time.time()
        return self.metrics.get_statistics()

class TestLoadPerformance:
    """负载性能测试"""
    
    @pytest.mark.asyncio
    async def test_experiment_creation_load(self):
        """测试实验创建负载"""
        tester = LoadTester()
        
        def data_generator(index):
            return {
                "name": f"负载测试实验_{index}",
                "description": "负载测试",
                "type": "A/B",
                "variants": [
                    {"name": "对照组", "traffic_percentage": 50, "is_control": True},
                    {"name": "实验组", "traffic_percentage": 50, "is_control": False}
                ],
                "metrics": [
                    {"name": "转化率", "type": "primary", "aggregation": "mean"}
                ],
                "sample_size": 1000,
                "confidence_level": 95
            }
        
        stats = await tester.run_concurrent_requests(
            "/api/v1/experiments",
            num_requests=100,
            concurrency=10,
            method="POST",
            data_generator=data_generator
        )
        
        # 断言性能指标
        assert stats["error_rate"] < 0.05  # 错误率小于5%
        assert stats["p95_response_time"] < 1.0  # 95分位响应时间小于1秒
        assert stats["throughput"] > 5  # 吞吐量大于5 req/s
        
        logger.info(f"实验创建性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_traffic_allocation_load(self):
        """测试流量分配负载"""
        tester = LoadTester()
        
        # 先创建一个实验
        experiment_id = "test_experiment_load"
        
        # 模拟大量用户请求分配
        async def allocate_users():
            tasks = []
            for i in range(1000):
                user_id = f"user_{i}"
                endpoint = f"/api/v1/traffic-allocation/assign?experiment_id={experiment_id}&user_id={user_id}"
                tasks.append(tester.make_request(None, endpoint))
            
            await asyncio.gather(*tasks)
        
        start_time = time.time()
        await allocate_users()
        duration = time.time() - start_time
        
        stats = tester.metrics.get_statistics()
        
        # 断言性能指标
        assert stats["error_rate"] < 0.01  # 错误率小于1%
        assert stats["mean_response_time"] < 0.1  # 平均响应时间小于100ms
        assert duration < 10  # 总时间小于10秒
        
        logger.info(f"流量分配性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_event_tracking_load(self):
        """测试事件跟踪负载"""
        tester = LoadTester()
        
        def event_generator(index):
            return {
                "events": [
                    {
                        "experiment_id": "test_experiment",
                        "user_id": f"user_{index % 100}",
                        "event_type": random.choice(["view", "click", "conversion"]),
                        "properties": {
                            "value": random.random() * 100,
                            "timestamp": utc_now().isoformat()
                        }
                    }
                    for _ in range(10)  # 每个批次10个事件
                ]
            }
        
        stats = await tester.run_concurrent_requests(
            "/api/v1/event-batch/track",
            num_requests=500,
            concurrency=50,
            method="POST",
            data_generator=event_generator
        )
        
        # 断言性能指标
        assert stats["error_rate"] < 0.02  # 错误率小于2%
        assert stats["p99_response_time"] < 2.0  # 99分位响应时间小于2秒
        assert stats["throughput"] > 20  # 吞吐量大于20 req/s
        
        logger.info(f"事件跟踪性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_metrics_calculation_load(self):
        """测试指标计算负载"""
        tester = LoadTester()
        
        # 并发请求实时指标
        experiment_ids = [f"exp_{i}" for i in range(10)]
        
        async def request_metrics():
            tasks = []
            for exp_id in experiment_ids:
                for _ in range(10):
                    endpoint = f"/api/v1/realtime-metrics/{exp_id}"
                    tasks.append(tester.make_request(None, endpoint))
            
            await asyncio.gather(*tasks)
        
        start_time = time.time()
        await request_metrics()
        duration = time.time() - start_time
        
        stats = tester.metrics.get_statistics()
        
        # 断言性能指标
        assert stats["mean_response_time"] < 0.5  # 平均响应时间小于500ms
        assert stats["p95_response_time"] < 1.0  # 95分位响应时间小于1秒
        
        logger.info(f"指标计算性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_load(self):
        """测试统计分析负载"""
        tester = LoadTester()
        
        def analysis_data_generator(index):
            # 生成随机数据进行统计分析
            control_data = np.random.normal(100, 15, 1000).tolist()
            treatment_data = np.random.normal(105, 15, 1000).tolist()
            
            return {
                "control_data": control_data,
                "treatment_data": treatment_data,
                "confidence_level": 0.95
            }
        
        stats = await tester.run_concurrent_requests(
            "/api/v1/statistical-analysis/t-test",
            num_requests=50,
            concurrency=5,
            method="POST",
            data_generator=analysis_data_generator
        )
        
        # 统计分析较慢，放宽限制
        assert stats["error_rate"] < 0.1  # 错误率小于10%
        assert stats["p95_response_time"] < 5.0  # 95分位响应时间小于5秒
        
        logger.info(f"统计分析性能统计: {json.dumps(stats, indent=2)}")

class TestConcurrencyScenarios:
    """并发场景测试"""
    
    @pytest.mark.asyncio
    async def test_mixed_workload(self):
        """测试混合工作负载"""
        tester = LoadTester()
        
        async def mixed_operations():
            tasks = []
            
            # 20% 创建实验
            for i in range(20):
                data = {
                    "name": f"实验_{i}",
                    "type": "A/B",
                    "variants": [
                        {"name": "对照组", "traffic_percentage": 50, "is_control": True},
                        {"name": "实验组", "traffic_percentage": 50, "is_control": False}
                    ]
                }
                tasks.append(tester.make_request(None, "/api/v1/experiments", "POST", data))
            
            # 50% 流量分配
            for i in range(50):
                endpoint = f"/api/v1/traffic-allocation/assign?experiment_id=exp1&user_id=user_{i}"
                tasks.append(tester.make_request(None, endpoint))
            
            # 30% 事件跟踪
            for i in range(30):
                data = {
                    "events": [{
                        "experiment_id": "exp1",
                        "user_id": f"user_{i}",
                        "event_type": "conversion"
                    }]
                }
                tasks.append(tester.make_request(None, "/api/v1/event-batch/track", "POST", data))
            
            await asyncio.gather(*tasks)
        
        tester.metrics.start_time = time.time()
        await mixed_operations()
        tester.metrics.end_time = time.time()
        
        stats = tester.metrics.get_statistics()
        
        # 混合负载的性能要求
        assert stats["error_rate"] < 0.05
        assert stats["mean_response_time"] < 1.0
        
        logger.info(f"混合负载性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_spike_load(self):
        """测试峰值负载"""
        tester = LoadTester()
        
        # 模拟突发流量
        async def spike_traffic():
            # 正常流量
            for _ in range(10):
                await tester.make_request(None, "/api/v1/experiments")
                await asyncio.sleep(0.1)
            
            # 突发流量
            tasks = []
            for i in range(100):
                endpoint = f"/api/v1/traffic-allocation/assign?experiment_id=exp1&user_id=spike_{i}"
                tasks.append(tester.make_request(None, endpoint))
            
            await asyncio.gather(*tasks)
            
            # 恢复正常
            for _ in range(10):
                await tester.make_request(None, "/api/v1/experiments")
                await asyncio.sleep(0.1)
        
        await spike_traffic()
        stats = tester.metrics.get_statistics()
        
        # 峰值负载下的性能要求
        assert stats["error_rate"] < 0.1
        assert stats["p99_response_time"] < 3.0
        
        logger.info(f"峰值负载性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """测试持续负载"""
        tester = LoadTester()
        duration_seconds = 30
        requests_per_second = 10
        
        async def sustained_traffic():
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < duration_seconds:
                tasks = []
                for _ in range(requests_per_second):
                    user_id = f"sustained_{request_count}"
                    endpoint = f"/api/v1/traffic-allocation/assign?experiment_id=exp1&user_id={user_id}"
                    tasks.append(tester.make_request(None, endpoint))
                    request_count += 1
                
                await asyncio.gather(*tasks)
                await asyncio.sleep(1)  # 每秒发送一批
        
        await sustained_traffic()
        stats = tester.metrics.get_statistics()
        
        # 持续负载下的性能要求
        assert stats["error_rate"] < 0.01
        assert stats["mean_response_time"] < 0.5
        
        logger.info(f"持续负载性能统计: {json.dumps(stats, indent=2)}")

class TestResourceOptimization:
    """资源优化测试"""
    
    @pytest.mark.asyncio
    async def test_database_connection_pooling(self):
        """测试数据库连接池"""
        tester = LoadTester()
        
        # 大量并发数据库操作
        async def db_intensive_operations():
            tasks = []
            for i in range(100):
                # 查询实验列表（涉及数据库查询）
                tasks.append(tester.make_request(None, "/api/v1/experiments"))
            
            await asyncio.gather(*tasks)
        
        await db_intensive_operations()
        stats = tester.metrics.get_statistics()
        
        # 数据库操作性能要求
        assert stats["error_rate"] < 0.01
        assert stats["mean_response_time"] < 0.2
        
        logger.info(f"数据库连接池性能统计: {json.dumps(stats, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self):
        """测试缓存有效性"""
        tester = LoadTester()
        
        # 重复请求相同数据
        experiment_id = "cached_experiment"
        
        # 第一轮请求（冷缓存）
        cold_stats = []
        for _ in range(10):
            start = time.time()
            await tester.make_request(None, f"/api/v1/experiments/{experiment_id}")
            cold_stats.append(time.time() - start)
        
        # 第二轮请求（热缓存）
        hot_stats = []
        for _ in range(10):
            start = time.time()
            await tester.make_request(None, f"/api/v1/experiments/{experiment_id}")
            hot_stats.append(time.time() - start)
        
        # 缓存应该显著提升性能
        cold_mean = statistics.mean(cold_stats)
        hot_mean = statistics.mean(hot_stats)
        
        assert hot_mean < cold_mean * 0.5  # 缓存后响应时间减少50%以上
        
        logger.info(f"缓存性能提升: {(1 - hot_mean/cold_mean) * 100:.2f}%")
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self):
        """测试批处理效率"""
        tester = LoadTester()
        
        # 单个事件 vs 批量事件
        single_events_time = 0
        batch_events_time = 0
        
        # 单个事件发送
        start = time.time()
        for i in range(100):
            data = {
                "events": [{
                    "experiment_id": "test",
                    "user_id": f"user_{i}",
                    "event_type": "view"
                }]
            }
            await tester.make_request(None, "/api/v1/event-batch/track", "POST", data)
        single_events_time = time.time() - start
        
        # 批量事件发送
        start = time.time()
        data = {
            "events": [
                {
                    "experiment_id": "test",
                    "user_id": f"user_{i}",
                    "event_type": "view"
                }
                for i in range(100)
            ]
        }
        await tester.make_request(None, "/api/v1/event-batch/track", "POST", data)
        batch_events_time = time.time() - start
        
        # 批处理应该更高效
        assert batch_events_time < single_events_time * 0.2
        
        logger.info(f"批处理效率提升: {(1 - batch_events_time/single_events_time) * 100:.2f}%")

class TestMemoryAndLeaks:
    """内存和泄漏测试"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        tester = LoadTester()
        
        # 执行大量请求
        for _ in range(5):
            await tester.run_concurrent_requests(
                "/api/v1/experiments",
                num_requests=100,
                concurrency=10
            )
        
        # 等待垃圾回收
        import gc
        gc.collect()
        await asyncio.sleep(2)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内
        assert memory_increase < 100  # 内存增长小于100MB
        
        logger.info(f"内存使用: 初始={initial_memory:.2f}MB, 最终={final_memory:.2f}MB, 增长={memory_increase:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self):
        """测试连接清理"""
        tester = LoadTester()
        
        # 创建大量连接
        async def create_connections():
            tasks = []
            for i in range(50):
                tasks.append(tester.make_request(None, "/api/v1/experiments"))
            await asyncio.gather(*tasks)
        
        # 执行多轮
        for _ in range(3):
            await create_connections()
            await asyncio.sleep(1)
        
        # 检查是否有连接泄漏（通过错误率判断）
        stats = tester.metrics.get_statistics()
        assert stats["error_rate"] < 0.05
        
        logger.error(f"连接清理测试: 错误率={stats['error_rate']:.2%}")

if __name__ == "__main__":
    setup_logging()
    # 运行性能测试
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # 运行基准测试
        asyncio.run(TestLoadPerformance().test_experiment_creation_load())
        asyncio.run(TestLoadPerformance().test_traffic_allocation_load())
        asyncio.run(TestLoadPerformance().test_event_tracking_load())
    else:
        # 运行pytest
        pytest.main([__file__, "-v"])
