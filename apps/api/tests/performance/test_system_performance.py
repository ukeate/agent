"""系统性能基准测试"""

import pytest
import asyncio
import aiohttp
import time
from typing import List, Dict, Any
from datetime import datetime
import statistics
import json
from concurrent.futures import ThreadPoolExecutor
import psutil
import tracemalloc


class PerformanceMetrics:
    """性能指标收集器"""
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
        
    def record_success(self, response_time: float):
        """记录成功请求"""
        self.response_times.append(response_time)
        self.success_count += 1
    
    def record_error(self):
        """记录错误"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.response_times:
            return {"error": "No successful requests"}
        
        sorted_times = sorted(self.response_times)
        total_requests = self.success_count + self.error_count
        duration = (self.end_time - self.start_time) if self.end_time else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0,
            "min_response_time": min(sorted_times),
            "max_response_time": max(sorted_times),
            "mean_response_time": statistics.mean(sorted_times),
            "median_response_time": statistics.median(sorted_times),
            "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
            "p99_response_time": sorted_times[int(len(sorted_times) * 0.99)],
            "requests_per_second": total_requests / duration if duration > 0 else 0,
            "duration_seconds": duration
        }


class SystemPerformanceTest:
    """系统性能测试"""
    
    @pytest.fixture
    def api_client(self):
        """创建API客户端"""
        return aiohttp.ClientSession()
    
    @pytest.fixture
    def base_url(self):
        """API基础URL"""
        return "http://localhost:8000/api/v1"
    
    @pytest.mark.asyncio
    async def test_single_request_baseline(self, api_client, base_url):
        """测试单请求基线性能"""
        metrics = PerformanceMetrics()
        
        async with api_client as session:
            # 预热请求
            await session.get(f"{base_url}/health")
            
            # 测试请求
            start = time.time()
            async with session.get(f"{base_url}/health") as response:
                await response.text()
                response_time = time.time() - start
            
            if response.status == 200:
                metrics.record_success(response_time)
            else:
                metrics.record_error()
        
        # 验证基线性能
        assert response_time < 0.1  # 健康检查应该在100ms内响应
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, api_client, base_url):
        """测试并发请求性能"""
        metrics = PerformanceMetrics()
        concurrent_users = 10
        requests_per_user = 10
        
        async def make_request(session):
            """发起单个请求"""
            start = time.time()
            try:
                async with session.get(f"{base_url}/health") as response:
                    await response.text()
                    response_time = time.time() - start
                    
                    if response.status == 200:
                        metrics.record_success(response_time)
                    else:
                        metrics.record_error()
            except Exception:
                metrics.record_error()
        
        async with api_client as session:
            metrics.start_time = time.time()
            
            # 创建并发任务
            tasks = []
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    tasks.append(make_request(session))
            
            # 执行并发请求
            await asyncio.gather(*tasks)
            
            metrics.end_time = time.time()
        
        # 获取统计
        stats = metrics.get_stats()
        
        # 验证性能指标
        assert stats["successful_requests"] > 0
        assert stats["error_rate"] < 0.1  # 错误率小于10%
        assert stats["p95_response_time"] < 0.5  # P95响应时间小于500ms
        assert stats["requests_per_second"] > 50  # RPS大于50
    
    @pytest.mark.asyncio
    async def test_agent_creation_performance(self, api_client, base_url):
        """测试智能体创建性能"""
        metrics = PerformanceMetrics()
        num_agents = 20
        
        async def create_agent(session, agent_id):
            """创建智能体"""
            start = time.time()
            
            payload = {
                "name": f"TestAgent_{agent_id}",
                "system_message": "Test agent for performance testing",
                "llm_config": {"model": "gpt-4o-mini", "temperature": 0.7}
            }
            
            try:
                async with session.post(
                    f"{base_url}/agents",
                    json=payload
                ) as response:
                    await response.text()
                    response_time = time.time() - start
                    
                    if response.status in [200, 201]:
                        metrics.record_success(response_time)
                        return await response.json()
                    else:
                        metrics.record_error()
                        return None
            except Exception:
                metrics.record_error()
                return None
        
        async with api_client as session:
            metrics.start_time = time.time()
            
            # 并发创建智能体
            tasks = [create_agent(session, i) for i in range(num_agents)]
            results = await asyncio.gather(*tasks)
            
            metrics.end_time = time.time()
        
        # 获取统计
        stats = metrics.get_stats()
        
        # 验证性能
        assert stats["successful_requests"] >= num_agents * 0.8  # 至少80%成功
        assert stats["mean_response_time"] < 2.0  # 平均响应时间小于2秒
    
    @pytest.mark.asyncio
    async def test_message_processing_throughput(self, api_client, base_url):
        """测试消息处理吞吐量"""
        metrics = PerformanceMetrics()
        num_messages = 100
        
        async def send_message(session, message_id):
            """发送消息"""
            start = time.time()
            
            payload = {
                "agent_id": "test_agent",
                "conversation_id": "test_conversation",
                "message": f"Test message {message_id}",
                "stream": False
            }
            
            try:
                async with session.post(
                    f"{base_url}/messages",
                    json=payload
                ) as response:
                    await response.text()
                    response_time = time.time() - start
                    
                    if response.status == 200:
                        metrics.record_success(response_time)
                    else:
                        metrics.record_error()
            except Exception:
                metrics.record_error()
        
        async with api_client as session:
            metrics.start_time = time.time()
            
            # 批量发送消息
            batch_size = 10
            for i in range(0, num_messages, batch_size):
                batch_tasks = [
                    send_message(session, j) 
                    for j in range(i, min(i + batch_size, num_messages))
                ]
                await asyncio.gather(*batch_tasks)
                await asyncio.sleep(0.1)  # 批次间短暂延迟
            
            metrics.end_time = time.time()
        
        # 获取统计
        stats = metrics.get_stats()
        
        # 验证吞吐量
        assert stats["requests_per_second"] > 10  # 至少10 RPS
        assert stats["p99_response_time"] < 5.0  # P99小于5秒
    
    @pytest.mark.asyncio
    async def test_websocket_connection_scaling(self, base_url):
        """测试WebSocket连接扩展性"""
        metrics = PerformanceMetrics()
        num_connections = 50
        ws_url = base_url.replace("http", "ws") + "/ws"
        
        async def create_ws_connection(connection_id):
            """创建WebSocket连接"""
            import websockets
            
            start = time.time()
            try:
                async with websockets.connect(ws_url) as websocket:
                    # 发送测试消息
                    await websocket.send(json.dumps({
                        "type": "test",
                        "connection_id": connection_id
                    }))
                    
                    # 接收响应
                    response = await websocket.recv()
                    
                    connection_time = time.time() - start
                    metrics.record_success(connection_time)
                    
                    # 保持连接一段时间
                    await asyncio.sleep(1)
                    
            except Exception:
                metrics.record_error()
        
        metrics.start_time = time.time()
        
        # 并发创建WebSocket连接
        tasks = [create_ws_connection(i) for i in range(num_connections)]
        await asyncio.gather(*tasks)
        
        metrics.end_time = time.time()
        
        # 获取统计
        stats = metrics.get_stats()
        
        # 验证WebSocket性能
        assert stats["successful_requests"] >= num_connections * 0.9  # 90%成功率
        assert stats["mean_response_time"] < 1.0  # 平均连接时间小于1秒
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """测试数据库查询性能"""
        from src.core.database import get_db
        
        metrics = PerformanceMetrics()
        num_queries = 100
        
        async def execute_query(query_id):
            """执行数据库查询"""
            start = time.time()
            
            try:
                db_gen = get_db()
                db = await anext(db_gen)
                try:
                    # 执行简单查询
                    result = await db.execute(
                        "SELECT id, name FROM agents LIMIT 10"
                    )
                    await result.fetchall()
                    
                    query_time = time.time() - start
                    metrics.record_success(query_time)
                finally:
                    # 确保生成器被正确关闭
                    try:
                        await anext(db_gen)
                    except StopAsyncIteration:
                        pass
                    
            except Exception:
                metrics.record_error()
        
        metrics.start_time = time.time()
        
        # 并发执行查询
        tasks = [execute_query(i) for i in range(num_queries)]
        await asyncio.gather(*tasks)
        
        metrics.end_time = time.time()
        
        # 获取统计
        stats = metrics.get_stats()
        
        # 验证数据库性能
        assert stats["mean_response_time"] < 0.05  # 平均查询时间小于50ms
        assert stats["p99_response_time"] < 0.2  # P99小于200ms
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """测试缓存命中率"""
        from src.core.redis import get_redis
        
        metrics = PerformanceMetrics()
        cache_keys = [f"test_key_{i}" for i in range(100)]
        
        redis = get_redis()
        
        # 预热缓存
        for key in cache_keys[:50]:  # 50%的key预先缓存
            await redis.set(key, f"value_{key}", ex=60)
        
        # 测试缓存访问
        cache_hits = 0
        cache_misses = 0
        
        for key in cache_keys:
            start = time.time()
            
            value = await redis.get(key)
            access_time = time.time() - start
            
            if value:
                cache_hits += 1
                metrics.record_success(access_time)
            else:
                cache_misses += 1
                # 模拟缓存未命中时的数据库查询
                await asyncio.sleep(0.01)
                metrics.record_success(access_time + 0.01)
        
        # 计算缓存命中率
        hit_rate = cache_hits / (cache_hits + cache_misses)
        
        # 验证缓存性能
        assert hit_rate >= 0.45  # 至少45%命中率（接近50%的预期）
        assert statistics.mean(metrics.response_times[:cache_hits]) < 0.001  # 缓存命中应该很快
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, api_client, base_url):
        """测试负载下的内存使用"""
        # 开始内存追踪
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 执行负载测试
        async with api_client as session:
            tasks = []
            for i in range(100):
                async def make_request():
                    async with session.get(f"{base_url}/health") as response:
                        await response.text()
                
                tasks.append(make_request())
            
            await asyncio.gather(*tasks)
        
        # 测量内存使用
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 获取内存快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]
        
        tracemalloc.stop()
        
        # 验证内存使用
        assert memory_increase < 100  # 内存增长小于100MB
        
        # 记录内存使用情况
        print(f"\\nMemory increase: {memory_increase:.2f} MB")
        print("Top memory allocations:")
        for stat in top_stats[:5]:
            print(stat)