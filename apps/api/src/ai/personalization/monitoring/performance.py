"""性能监控和优化模块"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Tuple, Deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import aioredis
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
recommendation_latency = Histogram(
    'personalization_recommendation_latency_seconds',
    'Recommendation request latency',
    ['scenario', 'cache_hit']
)

feature_computation_latency = Histogram(
    'personalization_feature_computation_latency_seconds',
    'Feature computation latency',
    ['feature_type']
)

model_inference_latency = Histogram(
    'personalization_model_inference_latency_seconds',
    'Model inference latency',
    ['model_type', 'version']
)

cache_hit_rate = Gauge(
    'personalization_cache_hit_rate',
    'Cache hit rate',
    ['cache_level']
)

active_users = Gauge(
    'personalization_active_users',
    'Number of active users'
)

recommendation_throughput = Summary(
    'personalization_throughput_rps',
    'Recommendations per second'
)

error_rate = Counter(
    'personalization_errors_total',
    'Total number of errors',
    ['error_type']
)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_size: int
    
@dataclass
class LatencyBreakdown:
    """延迟分解"""
    total: float
    feature_extraction: float
    model_inference: float
    cache_lookup: float
    post_processing: float
    network: float

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.latency_buffer: Deque[float] = deque(maxlen=10000)
        self.throughput_buffer: Deque[Tuple[datetime, int]] = deque(maxlen=1000)
        self.error_buffer: Deque[Tuple[datetime, str]] = deque(maxlen=1000)
        
        # 初始化OpenTelemetry
        self._init_tracing()
        
        # 性能阈值
        self.latency_threshold_p99 = 100  # 100ms
        self.error_rate_threshold = 0.01  # 1%
        self.cpu_threshold = 80  # 80%
        self.memory_threshold = 85  # 85%
        
    def _init_tracing(self):
        """初始化分布式追踪"""
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # 配置Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
    async def record_latency(self, 
                            latency: float,
                            scenario: str,
                            cache_hit: bool = False):
        """记录延迟"""
        self.latency_buffer.append(latency)
        recommendation_latency.labels(
            scenario=scenario,
            cache_hit=str(cache_hit)
        ).observe(latency)
        
        # 存储到Redis用于历史分析
        await self.redis.zadd(
            f"latency:{scenario}:{utc_now().strftime('%Y%m%d')}",
            {str(time.time()): latency}
        )
        
    async def record_throughput(self, count: int = 1):
        """记录吞吐量"""
        now = utc_now()
        self.throughput_buffer.append((now, count))
        recommendation_throughput.observe(count)
        
    async def record_error(self, error_type: str):
        """记录错误"""
        now = utc_now()
        self.error_buffer.append((now, error_type))
        error_rate.labels(error_type=error_type).inc()
        
    async def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        # 计算延迟百分位数
        latencies = list(self.latency_buffer)
        if latencies:
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
        else:
            p50 = p95 = p99 = 0
            
        # 计算吞吐量
        now = utc_now()
        recent_throughput = [
            count for timestamp, count in self.throughput_buffer
            if now - timestamp < timedelta(seconds=60)
        ]
        throughput = sum(recent_throughput) / 60 if recent_throughput else 0
        
        # 计算错误率
        recent_errors = [
            1 for timestamp, _ in self.error_buffer
            if now - timestamp < timedelta(minutes=5)
        ]
        total_requests = len(latencies)
        error_rate_value = len(recent_errors) / max(total_requests, 1)
        
        # 获取系统资源使用情况
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # 获取缓存命中率
        cache_stats = await self._get_cache_stats()
        
        return PerformanceMetrics(
            timestamp=now,
            p50_latency=p50,
            p95_latency=p95,
            p99_latency=p99,
            throughput=throughput,
            error_rate=error_rate_value,
            cache_hit_rate=cache_stats.get('hit_rate', 0),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=await self._get_active_connections(),
            queue_size=await self._get_queue_size()
        )
        
    async def _get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        hits = await self.redis.get("cache:hits") or 0
        misses = await self.redis.get("cache:misses") or 0
        total = int(hits) + int(misses)
        
        if total > 0:
            hit_rate = int(hits) / total
        else:
            hit_rate = 0
            
        cache_hit_rate.labels(cache_level="L2").set(hit_rate)
        
        return {
            'hits': int(hits),
            'misses': int(misses),
            'hit_rate': hit_rate
        }
        
    async def _get_active_connections(self) -> int:
        """获取活跃连接数"""
        # 从Redis获取活跃用户数
        active_users_count = await self.redis.scard("active_users")
        active_users.set(active_users_count)
        return active_users_count
        
    async def _get_queue_size(self) -> int:
        """获取队列大小"""
        return await self.redis.llen("recommendation_queue")
        
    async def check_performance_alerts(self) -> List[str]:
        """检查性能告警"""
        alerts = []
        metrics = await self.get_current_metrics()
        
        # 延迟告警
        if metrics.p99_latency > self.latency_threshold_p99:
            alerts.append(
                f"P99延迟过高: {metrics.p99_latency:.2f}ms > {self.latency_threshold_p99}ms"
            )
            
        # 错误率告警
        if metrics.error_rate > self.error_rate_threshold:
            alerts.append(
                f"错误率过高: {metrics.error_rate:.2%} > {self.error_rate_threshold:.2%}"
            )
            
        # CPU告警
        if metrics.cpu_usage > self.cpu_threshold:
            alerts.append(
                f"CPU使用率过高: {metrics.cpu_usage:.1f}% > {self.cpu_threshold}%"
            )
            
        # 内存告警
        if metrics.memory_usage > self.memory_threshold:
            alerts.append(
                f"内存使用率过高: {metrics.memory_usage:.1f}% > {self.memory_threshold}%"
            )
            
        # 队列积压告警
        if metrics.queue_size > 1000:
            alerts.append(
                f"队列积压严重: {metrics.queue_size} 个任务待处理"
            )
            
        return alerts

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def auto_optimize(self) -> Dict[str, Any]:
        """自动性能优化"""
        metrics = await self.monitor.get_current_metrics()
        optimizations = {}
        
        # 根据性能指标自动调整
        if metrics.p99_latency > 100:  # 延迟过高
            optimizations['cache_ttl'] = await self._optimize_cache_ttl()
            optimizations['batch_size'] = await self._optimize_batch_size()
            
        if metrics.error_rate > 0.01:  # 错误率过高
            optimizations['circuit_breaker'] = await self._adjust_circuit_breaker()
            optimizations['timeout'] = await self._adjust_timeout()
            
        if metrics.cpu_usage > 80:  # CPU使用率过高
            optimizations['worker_threads'] = await self._adjust_worker_threads()
            optimizations['model_complexity'] = await self._reduce_model_complexity()
            
        if metrics.memory_usage > 85:  # 内存使用率过高
            optimizations['cache_size'] = await self._adjust_cache_size()
            optimizations['batch_size'] = await self._reduce_batch_size()
            
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': utc_now(),
            'metrics': metrics,
            'optimizations': optimizations
        })
        
        return optimizations
        
    async def _optimize_cache_ttl(self) -> int:
        """优化缓存TTL"""
        # 根据缓存命中率动态调整TTL
        cache_stats = await self.monitor._get_cache_stats()
        hit_rate = cache_stats['hit_rate']
        
        if hit_rate < 0.5:
            return 300  # 5分钟
        elif hit_rate < 0.7:
            return 600  # 10分钟
        else:
            return 1800  # 30分钟
            
    async def _optimize_batch_size(self) -> int:
        """优化批处理大小"""
        metrics = await self.monitor.get_current_metrics()
        
        if metrics.p99_latency > 150:
            return 16  # 减小批处理大小
        elif metrics.p99_latency < 50:
            return 64  # 增大批处理大小
        else:
            return 32  # 默认大小
            
    async def _adjust_circuit_breaker(self) -> Dict[str, Any]:
        """调整熔断器参数"""
        return {
            'failure_threshold': 5,
            'recovery_timeout': 30,
            'expected_exception': TimeoutError
        }
        
    async def _adjust_timeout(self) -> float:
        """调整超时时间"""
        metrics = await self.monitor.get_current_metrics()
        
        # 基于P99延迟动态调整超时
        return min(metrics.p99_latency * 3, 500) / 1000  # 转换为秒
        
    async def _adjust_worker_threads(self) -> int:
        """调整工作线程数"""
        cpu_count = psutil.cpu_count()
        metrics = await self.monitor.get_current_metrics()
        
        if metrics.cpu_usage > 80:
            return max(cpu_count - 2, 2)
        else:
            return cpu_count
            
    async def _reduce_model_complexity(self) -> str:
        """降低模型复杂度"""
        # 切换到更轻量的模型
        return "light_model_v1"
        
    async def _adjust_cache_size(self) -> int:
        """调整缓存大小"""
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 85:
            return 1000  # 减小缓存
        elif memory_usage < 50:
            return 10000  # 增大缓存
        else:
            return 5000  # 默认大小
            
    async def _reduce_batch_size(self) -> int:
        """减小批处理大小以降低内存使用"""
        return 16

class LoadTester:
    """负载测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        
    async def run_load_test(self,
                           duration: int = 60,
                           rps: int = 100,
                           scenario: str = "homepage") -> Dict[str, Any]:
        """运行负载测试"""
        import aiohttp
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                tasks = []
                
                # 创建批量请求
                for _ in range(rps // 10):  # 每100ms发送一批
                    task = self._make_request(session, scenario)
                    tasks.append(task)
                    
                # 执行请求
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 统计结果
                for result in results:
                    if isinstance(result, Exception):
                        failed_requests += 1
                    else:
                        successful_requests += 1
                        latencies.append(result['latency'])
                        
                # 控制发送速率
                await asyncio.sleep(0.1)
                
        # 计算统计指标
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / max(total_requests, 1)
        
        if latencies:
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
            
        test_result = {
            'duration': duration,
            'target_rps': rps,
            'actual_rps': total_requests / duration,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'p50_latency': p50_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'scenario': scenario
        }
        
        self.results.append(test_result)
        return test_result
        
    async def _make_request(self, 
                          session: 'aiohttp.ClientSession',
                          scenario: str) -> Dict[str, Any]:
        """发送测试请求"""
        import aiohttp
        
        url = f"{self.base_url}/api/v1/personalization/recommend"
        
        payload = {
            "user_id": f"test_user_{np.random.randint(1000)}",
            "scenario": scenario,
            "n_recommendations": 10,
            "context": {
                "device": "mobile",
                "location": "beijing",
                "time_of_day": "evening"
            }
        }
        
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload) as response:
                await response.json()
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                
                return {
                    'status': response.status,
                    'latency': latency,
                    'success': response.status == 200
                }
        except Exception as e:
            raise e
            
    async def stress_test(self) -> Dict[str, Any]:
        """压力测试 - 逐步增加负载直到系统崩溃"""
        max_rps = 1000
        step = 50
        current_rps = 50
        
        stress_results = []
        
        while current_rps <= max_rps:
            logger.info(f"Testing at {current_rps} RPS...")
            
            result = await self.run_load_test(
                duration=30,
                rps=current_rps,
                scenario="stress_test"
            )
            
            stress_results.append(result)
            
            # 如果成功率低于95%或P99延迟超过500ms，停止测试
            if result['success_rate'] < 0.95 or result['p99_latency'] > 500:
                logger.warning(f"System degradation detected at {current_rps} RPS")
                break
                
            current_rps += step
            
        return {
            'max_sustainable_rps': current_rps - step,
            'results': stress_results
        }
        
    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "No test results available"
            
        report = ["\n=== 负载测试报告 ==="]
        
        for result in self.results:
            report.append(f"\n场景: {result['scenario']}")
            report.append(f"持续时间: {result['duration']}秒")
            report.append(f"目标RPS: {result['target_rps']}")
            report.append(f"实际RPS: {result['actual_rps']:.2f}")
            report.append(f"总请求数: {result['total_requests']}")
            report.append(f"成功率: {result['success_rate']:.2%}")
            report.append(f"平均延迟: {result['avg_latency']:.2f}ms")
            report.append(f"P50延迟: {result['p50_latency']:.2f}ms")
            report.append(f"P95延迟: {result['p95_latency']:.2f}ms")
            report.append(f"P99延迟: {result['p99_latency']:.2f}ms")
            report.append("-" * 40)
            
        return "\n".join(report)