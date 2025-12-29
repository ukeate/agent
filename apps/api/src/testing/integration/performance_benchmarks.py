"""性能基准与压测模块"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import asyncio
import hashlib
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Dict, List, Optional
import httpx
import psutil
from sqlalchemy import text
from src.core.database import get_db_session
from src.core.monitoring import monitor
from src.core.utils.timezone_utils import utc_now

@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_qps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
@dataclass
class BenchmarkComparison:
    """基准对比结果"""
    metric_name: str
    before_value: float
    after_value: float
    improvement_percent: float
    target_improvement: float
    target_achieved: bool

class PerformanceBenchmarkSuite:
    """性能基准测试套件"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    async def run_comprehensive_benchmark(self, benchmark_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """运行基准测试（无静态假数据）"""
        types = benchmark_types or ["cpu", "memory", "io", "network", "database"]
        monitor.log_info(f"开始运行性能基准测试: {types}")

        results: Dict[str, Any] = {"timestamp": utc_now().isoformat(), "benchmarks": {}}
        for name in types:
            metrics = await self.run_benchmark(name)
            results["benchmarks"][name] = asdict(metrics)
        return results
        
    async def run_benchmark(self, benchmark_name: str) -> BenchmarkMetrics:
        if benchmark_name == "cpu":
            return await self.benchmark_cpu()
        if benchmark_name == "memory":
            return await self.benchmark_memory()
        if benchmark_name == "io":
            return await self.benchmark_io()
        if benchmark_name == "network":
            return await self.benchmark_network()
        if benchmark_name == "database":
            return await self.benchmark_database()
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
    def _measure_process(self) -> tuple[psutil.Process, float, float]:
        p = psutil.Process()
        cpu = p.cpu_times()
        cpu_time = float(getattr(cpu, "user", 0.0)) + float(getattr(cpu, "system", 0.0))
        mem_mb = p.memory_info().rss / 1024 / 1024
        return p, cpu_time, mem_mb

    def _finish_metrics(
        self,
        latencies_ms: List[float],
        total_ops: int,
        duration_s: float,
        errors: int,
        start_cpu_time: float,
        start_mem_mb: float,
        end_cpu_time: float,
        end_mem_mb: float,
    ) -> BenchmarkMetrics:
        latencies_ms.sort()
        p50 = latencies_ms[int(len(latencies_ms) * 0.5)] if latencies_ms else 0.0
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0
        p99 = latencies_ms[int(len(latencies_ms) * 0.99)] if latencies_ms else 0.0
        throughput = (total_ops - errors) / duration_s if duration_s > 0 else 0.0
        error_rate = errors / total_ops if total_ops > 0 else 1.0

        cpu_time_delta = max(0.0, end_cpu_time - start_cpu_time)
        cpu_percent = (
            cpu_time_delta / duration_s / (psutil.cpu_count() or 1) * 100 if duration_s > 0 else 0.0
        )

        return BenchmarkMetrics(
            latency_p50=float(p50),
            latency_p95=float(p95),
            latency_p99=float(p99),
            throughput_qps=float(throughput),
            error_rate=float(error_rate),
            memory_usage_mb=float(max(start_mem_mb, end_mem_mb)),
            cpu_usage_percent=float(cpu_percent),
        )

    async def benchmark_cpu(self) -> BenchmarkMetrics:
        latencies: List[float] = []
        errors = 0
        total_ops = 300

        _, cpu_start, mem_start = self._measure_process()
        t0 = time.perf_counter()
        for _ in range(total_ops):
            s = time.perf_counter()
            hashlib.sha256(os.urandom(2048)).digest()
            latencies.append((time.perf_counter() - s) * 1000)
        duration = time.perf_counter() - t0
        _, cpu_end, mem_end = self._measure_process()

        return self._finish_metrics(latencies, total_ops, duration, errors, cpu_start, mem_start, cpu_end, mem_end)

    async def benchmark_memory(self) -> BenchmarkMetrics:
        latencies: List[float] = []
        errors = 0
        total_ops = 120
        size = 1024 * 1024

        _, cpu_start, mem_start = self._measure_process()
        t0 = time.perf_counter()
        for _ in range(total_ops):
            s = time.perf_counter()
            buf = bytearray(size)
            buf[0] = 1
            buf[-1] = 1
            del buf
            latencies.append((time.perf_counter() - s) * 1000)
        duration = time.perf_counter() - t0
        _, cpu_end, mem_end = self._measure_process()

        return self._finish_metrics(latencies, total_ops, duration, errors, cpu_start, mem_start, cpu_end, mem_end)

    async def benchmark_io(self) -> BenchmarkMetrics:
        latencies: List[float] = []
        errors = 0
        total_ops = 20
        payload = os.urandom(256 * 1024)

        _, cpu_start, mem_start = self._measure_process()
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "io_benchmark.bin"
            for _ in range(total_ops):
                s = time.perf_counter()
                try:
                    path.write_bytes(payload)
                    _ = path.read_bytes()
                except Exception:
                    errors += 1
                latencies.append((time.perf_counter() - s) * 1000)
        duration = time.perf_counter() - t0
        _, cpu_end, mem_end = self._measure_process()

        return self._finish_metrics(latencies, total_ops, duration, errors, cpu_start, mem_start, cpu_end, mem_end)

    async def benchmark_network(self) -> BenchmarkMetrics:
        url = f"{self.base_url}/api/v1/health"
        total_ops = 50
        concurrency = 10
        errors = 0
        latencies: List[float] = []

        sem = asyncio.Semaphore(concurrency)

        async def _one(client: httpx.AsyncClient) -> None:
            nonlocal errors
            async with sem:
                s = time.perf_counter()
                try:
                    resp = await client.get(url, timeout=10.0)
                    resp.raise_for_status()
                except Exception:
                    errors += 1
                latencies.append((time.perf_counter() - s) * 1000)

        _, cpu_start, mem_start = self._measure_process()
        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[_one(client) for _ in range(total_ops)])
        duration = time.perf_counter() - t0
        _, cpu_end, mem_end = self._measure_process()

        return self._finish_metrics(latencies, total_ops, duration, errors, cpu_start, mem_start, cpu_end, mem_end)

    async def benchmark_database(self) -> BenchmarkMetrics:
        latencies: List[float] = []
        errors = 0
        total_ops = 80

        _, cpu_start, mem_start = self._measure_process()
        t0 = time.perf_counter()
        try:
            async with get_db_session() as session:
                for _ in range(total_ops):
                    s = time.perf_counter()
                    try:
                        r = await session.execute(text("SELECT 1"))
                        r.scalar_one()
                    except Exception:
                        errors += 1
                    latencies.append((time.perf_counter() - s) * 1000)
        except Exception:
            errors = total_ops
        duration = time.perf_counter() - t0
        _, cpu_end, mem_end = self._measure_process()

        return self._finish_metrics(latencies, total_ops, duration, errors, cpu_start, mem_start, cpu_end, mem_end)
        
    def compare_metrics(self, baseline: BenchmarkMetrics, current: BenchmarkMetrics) -> List[BenchmarkComparison]:
        """与上一轮结果对比"""
        comparisons = []
        def _safe_improvement(before: float, after: float, higher_better: bool) -> float:
            if not before:
                return 0.0
            if higher_better:
                return ((after - before) / before) * 100
            return ((before - after) / before) * 100
        
        # 延迟对比 (越低越好)
        comparisons.append(BenchmarkComparison(
            metric_name='latency_p50',
            before_value=baseline.latency_p50,
            after_value=current.latency_p50,
            improvement_percent=_safe_improvement(baseline.latency_p50, current.latency_p50, False),
            target_improvement=50,
            target_achieved=current.latency_p50 <= baseline.latency_p50 * 0.5
        ))
        
        comparisons.append(BenchmarkComparison(
            metric_name='latency_p95',
            before_value=baseline.latency_p95,
            after_value=current.latency_p95,
            improvement_percent=_safe_improvement(baseline.latency_p95, current.latency_p95, False),
            target_improvement=50,
            target_achieved=current.latency_p95 <= baseline.latency_p95 * 0.5
        ))
        
        # 吞吐量对比 (越高越好)
        comparisons.append(BenchmarkComparison(
            metric_name='throughput_qps',
            before_value=baseline.throughput_qps,
            after_value=current.throughput_qps,
            improvement_percent=_safe_improvement(baseline.throughput_qps, current.throughput_qps, True),
            target_improvement=100,  # 翻倍
            target_achieved=current.throughput_qps >= baseline.throughput_qps * 2
        ))
        
        # 错误率对比 (越低越好)
        comparisons.append(BenchmarkComparison(
            metric_name='error_rate',
            before_value=baseline.error_rate,
            after_value=current.error_rate,
            improvement_percent=((baseline.error_rate - current.error_rate) / baseline.error_rate) * 100 if baseline.error_rate > 0 else 0,
            target_improvement=50,
            target_achieved=current.error_rate <= baseline.error_rate * 0.5
        ))
        
        # 内存使用对比 (越低越好)
        comparisons.append(BenchmarkComparison(
            metric_name='memory_usage_mb',
            before_value=baseline.memory_usage_mb,
            after_value=current.memory_usage_mb,
            improvement_percent=_safe_improvement(baseline.memory_usage_mb, current.memory_usage_mb, False),
            target_improvement=25,
            target_achieved=current.memory_usage_mb <= baseline.memory_usage_mb * 0.75
        ))
        
        # CPU使用对比 (越低越好)
        comparisons.append(BenchmarkComparison(
            metric_name='cpu_usage_percent',
            before_value=baseline.cpu_usage_percent,
            after_value=current.cpu_usage_percent,
            improvement_percent=_safe_improvement(baseline.cpu_usage_percent, current.cpu_usage_percent, False),
            target_improvement=30,
            target_achieved=current.cpu_usage_percent <= baseline.cpu_usage_percent * 0.7
        ))
        
        return comparisons
        
    def generate_summary(self, all_comparisons: Dict[str, List[BenchmarkComparison]]) -> Dict[str, Any]:
        """生成性能基准测试汇总"""
        total_metrics = 0
        targets_achieved = 0
        
        avg_improvements = {
            'latency': [],
            'throughput': [],
            'error_rate': [],
            'resource_usage': []
        }
        
        for benchmark_comparisons in all_comparisons.values():
            for comparison in benchmark_comparisons:
                total_metrics += 1
                if comparison.target_achieved:
                    targets_achieved += 1
                    
                # 分类统计改进
                if 'latency' in comparison.metric_name:
                    avg_improvements['latency'].append(comparison.improvement_percent)
                elif 'throughput' in comparison.metric_name:
                    avg_improvements['throughput'].append(comparison.improvement_percent)
                elif 'error' in comparison.metric_name:
                    avg_improvements['error_rate'].append(comparison.improvement_percent)
                else:
                    avg_improvements['resource_usage'].append(comparison.improvement_percent)
                    
        return {
            'total_metrics_tested': total_metrics,
            'targets_achieved': targets_achieved,
            'success_rate': (targets_achieved / total_metrics * 100) if total_metrics > 0 else 0,
            'average_improvements': {
                category: statistics.mean(values) if values else 0
                for category, values in avg_improvements.items()
            },
            'epic5_objectives_met': {
                'response_time_50_percent_improvement': avg_improvements['latency'] and statistics.mean(avg_improvements['latency']) >= 50,
                'throughput_doubled': avg_improvements['throughput'] and statistics.mean(avg_improvements['throughput']) >= 100,
                'resource_efficiency_25_percent': avg_improvements['resource_usage'] and statistics.mean(avg_improvements['resource_usage']) >= 25
            }
        }

class LoadTestRunner:
    """负载测试运行器"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    async def run_load_test(
        self,
        target_qps: int = 100,
        duration_seconds: int = 30,
        concurrency: int = 20,
        path: str = "/api/v1/health",
    ) -> Dict[str, Any]:
        """运行负载测试（基于真实HTTP请求）"""
        url = f"{self.base_url}{path}"
        monitor.log_info(f"开始负载测试: target_qps={target_qps}, duration_seconds={duration_seconds}, concurrency={concurrency}")

        sem = asyncio.Semaphore(max(1, concurrency))
        latencies: List[float] = []
        errors = 0
        started = time.perf_counter()
        deadline = started + max(1, duration_seconds)
        interval = max(0.001, concurrency / max(1, target_qps))

        async def _worker(client: httpx.AsyncClient) -> None:
            nonlocal errors
            next_at = time.perf_counter()
            while True:
                now = time.perf_counter()
                if now >= deadline:
                    return
                next_at = max(next_at + interval, now)
                await asyncio.sleep(max(0.0, next_at - now))
                async with sem:
                    s = time.perf_counter()
                    try:
                        r = await client.get(url, timeout=10.0)
                        r.raise_for_status()
                    except Exception:
                        errors += 1
                    latencies.append((time.perf_counter() - s) * 1000)

        async with httpx.AsyncClient() as client:
            workers = [asyncio.create_task(_worker(client)) for _ in range(max(1, concurrency))]
            await asyncio.gather(*workers)

        duration = max(0.001, time.perf_counter() - started)
        total = len(latencies)
        latencies.sort()

        avg = sum(latencies) / total if total else 0.0
        p95 = latencies[int(total * 0.95)] if total else 0.0
        achieved_qps = (total - errors) / duration

        return {
            "timestamp": utc_now().isoformat(),
            "duration_seconds": int(duration_seconds),
            "target_qps": int(target_qps),
            "concurrency": int(concurrency),
            "total_requests": total,
            "successful_requests": total - errors,
            "failed_requests": errors,
            "error_rate": errors / total if total else 1.0,
            "avg_response_time_ms": avg,
            "p95_response_time_ms": p95,
            "achieved_qps": achieved_qps,
        }

class StressTestRunner:
    """压力测试运行器"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    async def run_stress_test(
        self,
        duration_seconds: int = 30,
        max_concurrency: int = 200,
        failure_threshold: float = 0.2,
        path: str = "/api/v1/health",
    ) -> Dict[str, Any]:
        """运行压力测试（逐步提升并发直到错误率超过阈值或达到上限）"""
        url = f"{self.base_url}{path}"
        monitor.log_info(
            f"开始运行压力测试: duration_seconds={duration_seconds}, max_concurrency={max_concurrency}, failure_threshold={failure_threshold}"
        )

        steps = [max(1, max_concurrency // 4), max(1, max_concurrency // 2), max_concurrency]
        stage_seconds = max(1, int(duration_seconds / len(steps)))
        stages = []

        async with httpx.AsyncClient() as client:
            for c in steps:
                stage = await self._run_stage(client, url, stage_seconds, c)
                stages.append(stage)
                if stage["error_rate"] >= failure_threshold:
                    break

        return {"timestamp": utc_now().isoformat(), "stages": stages, "failure_threshold": failure_threshold}

    async def _run_stage(
        self, client: httpx.AsyncClient, url: str, duration_seconds: int, concurrency: int
    ) -> Dict[str, Any]:
        sem = asyncio.Semaphore(max(1, concurrency))
        latencies: List[float] = []
        errors = 0
        started = time.perf_counter()
        deadline = started + max(1, duration_seconds)

        async def _one() -> None:
            nonlocal errors
            async with sem:
                s = time.perf_counter()
                try:
                    r = await client.get(url, timeout=10.0)
                    r.raise_for_status()
                except Exception:
                    errors += 1
                latencies.append((time.perf_counter() - s) * 1000)

        while time.perf_counter() < deadline:
            await asyncio.gather(*[_one() for _ in range(max(1, concurrency))])

        duration = max(0.001, time.perf_counter() - started)
        total = len(latencies)
        latencies.sort()
        p95 = latencies[int(total * 0.95)] if total else 0.0

        return {
            "concurrency": int(concurrency),
            "duration_seconds": int(duration_seconds),
            "total_requests": total,
            "failed_requests": errors,
            "error_rate": errors / total if total else 1.0,
            "p95_response_time_ms": p95,
            "achieved_qps": (total - errors) / duration,
        }
