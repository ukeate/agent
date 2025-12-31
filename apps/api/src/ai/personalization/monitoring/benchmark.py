"""性能基准测试模块"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ..engine import PersonalizationEngine
from ..features.realtime import RealTimeFeatureEngine
from ..models.service import ModelService
from ..cache.feature_cache import FeatureCache
from .performance import LoadTester, PerformanceMonitor
from src.core.logging import setup_logging

logger = get_logger(__name__)

@dataclass
class BenchmarkConfig:
    """基准测试配置"""

    name: str
    description: str
    duration: int = 60
    warmup_duration: int = 10
    user_count: int = 1000
    rps_targets: List[int] = None
    scenarios: List[str] = None
    feature_dimensions: int = 100
    model_complexity: str = "medium"
    cache_enabled: bool = True
    batch_sizes: List[int] = None
    
    def __post_init__(self):
        if self.rps_targets is None:
            self.rps_targets = [10, 50, 100, 200, 500]
        if self.scenarios is None:
            self.scenarios = ["homepage", "search", "category", "detail"]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 16, 32, 64]

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    config: BenchmarkConfig
    timestamp: datetime
    metrics: Dict[str, Any]
    latency_percentiles: Dict[str, float]
    throughput_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, 
                 engine: PersonalizationEngine,
                 monitor: PerformanceMonitor,
                 output_dir: str = "./benchmark_results"):
        self.engine = engine
        self.monitor = monitor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """运行基准测试"""
        logger.info(f"Starting benchmark: {config.name}")
        
        # 预热阶段
        await self._warmup(config)
        
        # 运行各种测试场景
        metrics = {}
        
        # 1. 延迟测试
        latency_results = await self._test_latency(config)
        metrics['latency'] = latency_results
        
        # 2. 吞吐量测试
        throughput_results = await self._test_throughput(config)
        metrics['throughput'] = throughput_results
        
        # 3. 并发测试
        concurrency_results = await self._test_concurrency(config)
        metrics['concurrency'] = concurrency_results
        
        # 4. 批处理测试
        batch_results = await self._test_batch_processing(config)
        metrics['batch_processing'] = batch_results
        
        # 5. 缓存效果测试
        cache_results = await self._test_cache_effectiveness(config)
        metrics['cache'] = cache_results
        
        # 6. 资源使用测试
        resource_results = await self._test_resource_usage(config)
        metrics['resources'] = resource_results
        
        # 分析瓶颈
        bottlenecks = self._identify_bottlenecks(metrics)
        
        # 生成优化建议
        recommendations = self._generate_recommendations(metrics, bottlenecks)
        
        # 计算关键指标
        latency_percentiles = self._calculate_latency_percentiles(latency_results)
        throughput_metrics = self._calculate_throughput_metrics(throughput_results)
        resource_usage = self._calculate_resource_usage(resource_results)
        
        result = BenchmarkResult(
            config=config,
            timestamp=utc_now(),
            metrics=metrics,
            latency_percentiles=latency_percentiles,
            throughput_metrics=throughput_metrics,
            resource_usage=resource_usage,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
        
        self.results.append(result)
        
        # 保存结果
        await self._save_results(result)
        
        # 生成报告
        await self._generate_report(result)
        
        return result
        
    async def _warmup(self, config: BenchmarkConfig):
        """预热阶段"""
        logger.info("Warming up...")
        
        # 预热缓存
        for i in range(100):
            user_id = f"warmup_user_{i}"
            request = {
                "user_id": user_id,
                "scenario": "homepage",
                "n_recommendations": 10,
                "context": {"warmup": True}
            }
            await self.engine.get_recommendations(request)
            
        # 等待系统稳定
        await asyncio.sleep(config.warmup_duration)
        
    async def _test_latency(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试延迟"""
        logger.info("Testing latency...")
        
        latency_results = {}
        
        for scenario in config.scenarios:
            latencies = []
            
            for i in range(100):  # 每个场景测试100次
                user_id = f"test_user_{i}"
                
                start_time = time.time()
                
                request = {
                    "user_id": user_id,
                    "scenario": scenario,
                    "n_recommendations": 10,
                    "context": {
                        "device": "mobile",
                        "location": "beijing"
                    }
                }
                
                await self.engine.get_recommendations(request)
                
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
                
            latency_results[scenario] = {
                'min': np.min(latencies),
                'max': np.max(latencies),
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'raw': latencies
            }
            
        return latency_results
        
    async def _test_throughput(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试吞吐量"""
        logger.info("Testing throughput...")
        
        throughput_results = {}
        
        for rps_target in config.rps_targets:
            tester = LoadTester()
            
            result = await tester.run_load_test(
                duration=config.duration,
                rps=rps_target,
                scenario="throughput_test"
            )
            
            throughput_results[f"{rps_target}_rps"] = {
                'actual_rps': result['actual_rps'],
                'success_rate': result['success_rate'],
                'avg_latency': result['avg_latency'],
                'p99_latency': result['p99_latency']
            }
            
        return throughput_results
        
    async def _test_concurrency(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试并发性能"""
        logger.info("Testing concurrency...")
        
        concurrency_levels = [1, 10, 50, 100, 200]
        concurrency_results = {}
        
        for level in concurrency_levels:
            start_time = time.time()
            
            tasks = []
            for i in range(level):
                user_id = f"concurrent_user_{i}"
                request = {
                    "user_id": user_id,
                    "scenario": "homepage",
                    "n_recommendations": 10,
                    "context": {}
                }
                task = self.engine.get_recommendations(request)
                tasks.append(task)
                
            await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            concurrency_results[f"level_{level}"] = {
                'total_time': duration,
                'avg_time_per_request': duration / level,
                'requests_per_second': level / duration
            }
            
        return concurrency_results
        
    async def _test_batch_processing(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试批处理性能"""
        logger.info("Testing batch processing...")
        
        batch_results = {}
        
        for batch_size in config.batch_sizes:
            requests = []
            
            for i in range(batch_size):
                user_id = f"batch_user_{i}"
                request = {
                    "user_id": user_id,
                    "scenario": "homepage",
                    "n_recommendations": 10,
                    "context": {}
                }
                requests.append(request)
                
            start_time = time.time()
            
            # 批量处理
            tasks = [self.engine.get_recommendations(req) for req in requests]
            await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            batch_results[f"batch_{batch_size}"] = {
                'total_time': duration,
                'avg_time_per_request': duration / batch_size,
                'throughput': batch_size / duration
            }
            
        return batch_results
        
    async def _test_cache_effectiveness(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试缓存效果"""
        logger.info("Testing cache effectiveness...")
        
        # 清空缓存
        await self.engine.feature_cache.clear()
        
        # 第一轮请求（冷缓存）
        cold_latencies = []
        for i in range(50):
            user_id = f"cache_test_user_{i % 10}"  # 重复用户
            
            start_time = time.time()
            
            request = {
                "user_id": user_id,
                "scenario": "homepage",
                "n_recommendations": 10,
                "context": {},
                "use_cache": True
            }
            
            await self.engine.get_recommendations(request)
            
            latency = (time.time() - start_time) * 1000
            cold_latencies.append(latency)
            
        # 第二轮请求（热缓存）
        hot_latencies = []
        for i in range(50):
            user_id = f"cache_test_user_{i % 10}"  # 相同用户
            
            start_time = time.time()
            
            request = {
                "user_id": user_id,
                "scenario": "homepage",
                "n_recommendations": 10,
                "context": {},
                "use_cache": True
            }
            
            await self.engine.get_recommendations(request)
            
            latency = (time.time() - start_time) * 1000
            hot_latencies.append(latency)
            
        cache_stats = await self.monitor._get_cache_stats()
        
        return {
            'cold_cache': {
                'avg_latency': np.mean(cold_latencies),
                'p99_latency': np.percentile(cold_latencies, 99)
            },
            'hot_cache': {
                'avg_latency': np.mean(hot_latencies),
                'p99_latency': np.percentile(hot_latencies, 99)
            },
            'improvement': {
                'latency_reduction': (np.mean(cold_latencies) - np.mean(hot_latencies)) / np.mean(cold_latencies),
                'hit_rate': cache_stats['hit_rate']
            }
        }
        
    async def _test_resource_usage(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """测试资源使用"""
        logger.info("Testing resource usage...")
        
        import psutil
        
        # 基准资源使用
        baseline_cpu = psutil.cpu_percent(interval=1)
        baseline_memory = psutil.virtual_memory().percent
        
        # 高负载测试
        tester = LoadTester()
        
        # 监控资源使用
        resource_samples = []
        
        async def monitor_resources():
            while True:
                resource_samples.append({
                    'timestamp': time.time(),
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes
                })
                await asyncio.sleep(1)
                
        # 启动监控
        monitor_task = create_task_with_logging(monitor_resources())
        
        # 运行负载测试
        await tester.run_load_test(
            duration=30,
            rps=200,
            scenario="resource_test"
        )
        
        # 停止监控
        monitor_task.cancel()
        
        # 分析资源使用
        cpu_usage = [s['cpu'] for s in resource_samples]
        memory_usage = [s['memory'] for s in resource_samples]
        
        return {
            'baseline': {
                'cpu': baseline_cpu,
                'memory': baseline_memory
            },
            'under_load': {
                'cpu_avg': np.mean(cpu_usage),
                'cpu_max': np.max(cpu_usage),
                'memory_avg': np.mean(memory_usage),
                'memory_max': np.max(memory_usage)
            },
            'samples': resource_samples
        }
        
    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 检查延迟瓶颈
        latency_data = metrics.get('latency', {})
        for scenario, stats in latency_data.items():
            if stats.get('p99', 0) > 200:  # P99 > 200ms
                bottlenecks.append(f"高延迟瓶颈: {scenario}场景P99延迟{stats['p99']:.2f}ms")
                
        # 检查吞吐量瓶颈
        throughput_data = metrics.get('throughput', {})
        for rps_config, stats in throughput_data.items():
            if stats.get('success_rate', 1) < 0.95:
                bottlenecks.append(f"吞吐量瓶颈: {rps_config}时成功率仅{stats['success_rate']:.2%}")
                
        # 检查资源瓶颈
        resource_data = metrics.get('resources', {})
        if resource_data.get('under_load', {}).get('cpu_max', 0) > 90:
            bottlenecks.append(f"CPU瓶颈: 峰值使用率{resource_data['under_load']['cpu_max']:.1f}%")
            
        if resource_data.get('under_load', {}).get('memory_max', 0) > 90:
            bottlenecks.append(f"内存瓶颈: 峰值使用率{resource_data['under_load']['memory_max']:.1f}%")
            
        # 检查缓存瓶颈
        cache_data = metrics.get('cache', {})
        if cache_data.get('improvement', {}).get('hit_rate', 0) < 0.5:
            bottlenecks.append(f"缓存效率低: 命中率仅{cache_data['improvement']['hit_rate']:.2%}")
            
        return bottlenecks
        
    def _generate_recommendations(self, 
                                 metrics: Dict[str, Any],
                                 bottlenecks: List[str]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于瓶颈生成建议
        for bottleneck in bottlenecks:
            if "高延迟" in bottleneck:
                recommendations.append("建议: 增加缓存层或优化特征计算逻辑")
                recommendations.append("建议: 考虑使用更快的模型或减少特征维度")
                
            elif "吞吐量" in bottleneck:
                recommendations.append("建议: 增加服务实例数或使用负载均衡")
                recommendations.append("建议: 优化批处理大小和并发配置")
                
            elif "CPU瓶颈" in bottleneck:
                recommendations.append("建议: 使用更高效的算法或数据结构")
                recommendations.append("建议: 考虑模型量化或剪枝")
                
            elif "内存瓶颈" in bottleneck:
                recommendations.append("建议: 减小缓存大小或使用LRU淘汰策略")
                recommendations.append("建议: 优化数据结构减少内存占用")
                
            elif "缓存效率低" in bottleneck:
                recommendations.append("建议: 增加缓存TTL或预热常用数据")
                recommendations.append("建议: 使用多级缓存策略")
                
        # 通用建议
        cache_data = metrics.get('cache', {})
        if cache_data.get('improvement', {}).get('latency_reduction', 0) > 0.5:
            recommendations.append("建议: 缓存效果显著，可考虑增加缓存容量")
            
        resource_data = metrics.get('resources', {})
        cpu_avg = resource_data.get('under_load', {}).get('cpu_avg', 0)
        if cpu_avg < 50:
            recommendations.append("建议: CPU使用率较低，可增加并发处理能力")
            
        return recommendations
        
    def _calculate_latency_percentiles(self, latency_results: Dict[str, Any]) -> Dict[str, float]:
        """计算延迟百分位数"""
        all_latencies = []
        
        for scenario_stats in latency_results.values():
            all_latencies.extend(scenario_stats.get('raw', []))
            
        if not all_latencies:
            return {}
            
        return {
            'p50': np.percentile(all_latencies, 50),
            'p75': np.percentile(all_latencies, 75),
            'p90': np.percentile(all_latencies, 90),
            'p95': np.percentile(all_latencies, 95),
            'p99': np.percentile(all_latencies, 99),
            'p999': np.percentile(all_latencies, 99.9)
        }
        
    def _calculate_throughput_metrics(self, throughput_results: Dict[str, Any]) -> Dict[str, float]:
        """计算吞吐量指标"""
        max_sustainable_rps = 0
        
        for rps_config, stats in throughput_results.items():
            if stats.get('success_rate', 0) >= 0.95 and stats.get('p99_latency', float('inf')) <= 200:
                rps_value = int(rps_config.split('_')[0])
                max_sustainable_rps = max(max_sustainable_rps, rps_value)
                
        actual_rps_values = [stats.get('actual_rps', 0) for stats in throughput_results.values()]
        
        return {
            'max_sustainable_rps': max_sustainable_rps,
            'avg_actual_rps': np.mean(actual_rps_values) if actual_rps_values else 0,
            'peak_actual_rps': np.max(actual_rps_values) if actual_rps_values else 0
        }
        
    def _calculate_resource_usage(self, resource_results: Dict[str, Any]) -> Dict[str, float]:
        """计算资源使用指标"""
        under_load = resource_results.get('under_load', {})
        baseline = resource_results.get('baseline', {})
        
        return {
            'cpu_baseline': baseline.get('cpu', 0),
            'cpu_avg': under_load.get('cpu_avg', 0),
            'cpu_max': under_load.get('cpu_max', 0),
            'memory_baseline': baseline.get('memory', 0),
            'memory_avg': under_load.get('memory_avg', 0),
            'memory_max': under_load.get('memory_max', 0)
        }
        
    async def _save_results(self, result: BenchmarkResult):
        """保存测试结果"""
        timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"benchmark_{result.config.name}_{timestamp}.json"
        
        # 转换为可序列化格式
        data = {
            'config': asdict(result.config),
            'timestamp': result.timestamp.isoformat(),
            'metrics': result.metrics,
            'latency_percentiles': result.latency_percentiles,
            'throughput_metrics': result.throughput_metrics,
            'resource_usage': result.resource_usage,
            'bottlenecks': result.bottlenecks,
            'recommendations': result.recommendations
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Results saved to {filename}")
        
    async def _generate_report(self, result: BenchmarkResult):
        """生成测试报告"""
        timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
        
        # 生成文本报告
        report_file = self.output_dir / f"report_{result.config.name}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"=== 性能基准测试报告 ===\n")
            f.write(f"测试名称: {result.config.name}\n")
            f.write(f"测试时间: {result.timestamp}\n")
            f.write(f"测试描述: {result.config.description}\n\n")
            
            f.write("=== 延迟指标 ===\n")
            for percentile, value in result.latency_percentiles.items():
                f.write(f"{percentile}: {value:.2f}ms\n")
                
            f.write("\n=== 吞吐量指标 ===\n")
            for metric, value in result.throughput_metrics.items():
                f.write(f"{metric}: {value:.2f}\n")
                
            f.write("\n=== 资源使用 ===\n")
            for metric, value in result.resource_usage.items():
                f.write(f"{metric}: {value:.2f}%\n")
                
            f.write("\n=== 性能瓶颈 ===\n")
            for bottleneck in result.bottlenecks:
                f.write(f"- {bottleneck}\n")
                
            f.write("\n=== 优化建议 ===\n")
            for recommendation in result.recommendations:
                f.write(f"- {recommendation}\n")
                
        # 生成可视化报告
        await self._generate_visualizations(result)
        
        logger.info(f"Report generated at {report_file}")
        
    async def _generate_visualizations(self, result: BenchmarkResult):
        """生成可视化图表"""
        timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        
        # 1. 延迟分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 场景延迟对比
        latency_data = result.metrics.get('latency', {})
        scenarios = list(latency_data.keys())
        p99_values = [latency_data[s].get('p99', 0) for s in scenarios]
        
        axes[0, 0].bar(scenarios, p99_values)
        axes[0, 0].set_title('P99延迟对比')
        axes[0, 0].set_ylabel('延迟 (ms)')
        axes[0, 0].set_xlabel('场景')
        
        # 吞吐量vs延迟
        throughput_data = result.metrics.get('throughput', {})
        rps_values = []
        latency_values = []
        
        for rps_config, stats in throughput_data.items():
            rps_values.append(int(rps_config.split('_')[0]))
            latency_values.append(stats.get('p99_latency', 0))
            
        axes[0, 1].plot(rps_values, latency_values, marker='o')
        axes[0, 1].set_title('吞吐量 vs 延迟')
        axes[0, 1].set_xlabel('RPS')
        axes[0, 1].set_ylabel('P99延迟 (ms)')
        
        # 资源使用趋势
        resource_data = result.metrics.get('resources', {})
        samples = resource_data.get('samples', [])
        
        if samples:
            timestamps = [s['timestamp'] for s in samples]
            cpu_values = [s['cpu'] for s in samples]
            memory_values = [s['memory'] for s in samples]
            
            axes[1, 0].plot(timestamps, cpu_values, label='CPU')
            axes[1, 0].plot(timestamps, memory_values, label='内存')
            axes[1, 0].set_title('资源使用趋势')
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('使用率 (%)')
            axes[1, 0].legend()
            
        # 缓存效果
        cache_data = result.metrics.get('cache', {})
        cold_latency = cache_data.get('cold_cache', {}).get('avg_latency', 0)
        hot_latency = cache_data.get('hot_cache', {}).get('avg_latency', 0)
        
        axes[1, 1].bar(['冷缓存', '热缓存'], [cold_latency, hot_latency])
        axes[1, 1].set_title('缓存效果')
        axes[1, 1].set_ylabel('平均延迟 (ms)')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / f"benchmark_charts_{result.config.name}_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {chart_file}")

async def run_comprehensive_benchmark():
    """运行综合性能基准测试"""
    # 初始化组件
    from ...core.config import get_settings
    import redis.asyncio as redis_async

    settings = get_settings()
    redis = redis_async.Redis.from_url(settings.REDIS_URL)
    
    feature_cache = FeatureCache(redis)
    feature_engine = RealTimeFeatureEngine(redis)
    model_service = ModelService(redis)
    
    engine = PersonalizationEngine(
        feature_engine=feature_engine,
        model_service=model_service,
        feature_cache=feature_cache,
        redis=redis
    )
    
    monitor = PerformanceMonitor(redis)
    
    benchmark = PerformanceBenchmark(engine, monitor)
    
    # 定义测试配置
    configs = [
        BenchmarkConfig(
            name="baseline",
            description="基准性能测试",
            duration=60,
            rps_targets=[10, 50, 100, 200],
            scenarios=["homepage", "search", "category"]
        ),
        BenchmarkConfig(
            name="stress",
            description="压力测试",
            duration=120,
            rps_targets=[100, 200, 500, 1000],
            cache_enabled=True
        ),
        BenchmarkConfig(
            name="cache_comparison",
            description="缓存效果对比",
            duration=60,
            cache_enabled=True
        )
    ]
    
    # 运行所有测试
    for config in configs:
        result = await benchmark.run_benchmark(config)
        logger.info(f"Benchmark {config.name} completed")
        
        # 打印关键结果
        logger.info("基准测试结果", name=config.name)
        logger.info("P99延迟", p99_ms=round(result.latency_percentiles.get("p99", 0), 2))
        logger.info(
            "最大可持续RPS",
            max_sustainable_rps=result.throughput_metrics.get("max_sustainable_rps", 0),
        )
        logger.info("CPU峰值", cpu_max=round(result.resource_usage.get("cpu_max", 0), 1))
        logger.info("瓶颈数", total=len(result.bottlenecks))
        
    # 清理
    await redis.aclose()

if __name__ == "__main__":
    setup_logging()
    asyncio.run(run_comprehensive_benchmark())
from src.core.logging import get_logger
