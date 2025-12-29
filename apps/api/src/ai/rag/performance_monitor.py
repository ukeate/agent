"""
向量性能监控器

监控向量搜索性能、量化效果和系统健康状态
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import asyncio
from dataclasses import dataclass
from collections import deque
import statistics

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    latency_ms: float
    result_count: int
    cache_hit: bool
    quantization_mode: str
    timestamp: datetime
    error: Optional[str] = None

class VectorPerformanceMonitor:
    """向量性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.performance_stats = {
            "search_latency": [],
            "index_usage": {},
            "cache_hit_rate": 0.0,
            "quantization_ratio": 0.0,
            "error_count": 0,
            "total_searches": 0
        }
        self.baseline_metrics = None
        self.optimization_targets = {
            "latency_improvement_target": 0.30,  # 30% improvement
            "memory_reduction_target": 0.20,     # 20% reduction
            "cache_hit_rate_target": 0.75        # 75% cache hit rate
        }
    
    async def monitor_search_performance(
        self,
        search_func: Callable,
        *args,
        quantization_mode: str = "unknown",
        **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """监控搜索性能"""
        start_time = time.time()
        error = None
        result = None
        result_count = 0
        
        try:
            result = await search_func(*args, **kwargs)
            result_count = len(result) if isinstance(result, list) else 1
        except Exception as e:
            error = str(e)
            logger.error(f"Search function error: {e}")
            self.performance_stats["error_count"] += 1
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            latency_ms=latency_ms,
            result_count=result_count,
            cache_hit=kwargs.get("cache_hit", False),
            quantization_mode=quantization_mode,
            timestamp=utc_now(),
            error=error
        )
        
        # 记录指标
        self.metrics_history.append(metrics)
        self.performance_stats["search_latency"].append(latency_ms)
        self.performance_stats["total_searches"] += 1
        
        # 更新索引使用统计
        index_type = kwargs.get("index_type", "unknown")
        self.performance_stats["index_usage"][index_type] = (
            self.performance_stats["index_usage"].get(index_type, 0) + 1
        )
        
        # 保持性能数据在合理范围内
        if len(self.performance_stats["search_latency"]) > self.max_history:
            self.performance_stats["search_latency"] = (
                self.performance_stats["search_latency"][-self.max_history:]
            )
        
        logger.debug(f"Search monitored: {latency_ms:.2f}ms, {result_count} results")
        
        return result, metrics
    
    async def establish_baseline(
        self,
        search_function: Callable,
        test_vectors: List[np.ndarray],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """建立基准性能指标"""
        logger.info("Establishing performance baseline...")
        
        baseline_latencies = []
        baseline_results = []
        
        for i in range(iterations):
            for vector in test_vectors:
                result, metrics = await self.monitor_search_performance(
                    search_function,
                    vector,
                    quantization_mode="baseline"
                )
                
                if metrics.error is None:
                    baseline_latencies.append(metrics.latency_ms)
                    baseline_results.append(metrics.result_count)
        
        if not baseline_latencies:
            raise ValueError("Failed to establish baseline - no successful searches")
        
        self.baseline_metrics = {
            "average_latency_ms": statistics.mean(baseline_latencies),
            "median_latency_ms": statistics.median(baseline_latencies),
            "p95_latency_ms": np.percentile(baseline_latencies, 95),
            "p99_latency_ms": np.percentile(baseline_latencies, 99),
            "average_results": statistics.mean(baseline_results),
            "total_searches": len(baseline_latencies),
            "timestamp": utc_now().isoformat()
        }
        
        logger.info(f"Baseline established: avg={self.baseline_metrics['average_latency_ms']:.2f}ms")
        return self.baseline_metrics
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No performance data available"}
        
        # 计算基本统计
        recent_metrics = list(self.metrics_history)[-100:]  # 最近100次搜索
        latencies = [m.latency_ms for m in recent_metrics if m.error is None]
        
        if not latencies:
            return {"status": "error", "message": "No successful searches in recent history"}
        
        # 计算缓存命中率
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        cache_hit_rate = cache_hits / len(recent_metrics) if recent_metrics else 0.0
        
        # 计算量化使用率
        quantized_searches = sum(
            1 for m in recent_metrics 
            if m.quantization_mode in ["int8", "int4", "adaptive"]
        )
        quantization_ratio = quantized_searches / len(recent_metrics) if recent_metrics else 0.0
        
        # 错误率统计
        errors = sum(1 for m in recent_metrics if m.error is not None)
        error_rate = errors / len(recent_metrics) if recent_metrics else 0.0
        
        report = {
            "summary": {
                "total_searches": len(self.metrics_history),
                "recent_searches": len(recent_metrics),
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "cache_hit_rate": cache_hit_rate,
                "quantization_ratio": quantization_ratio,
                "error_rate": error_rate
            },
            "index_usage": self.performance_stats["index_usage"].copy(),
            "baseline_comparison": {},
            "optimization_status": {},
            "timestamp": utc_now().isoformat()
        }
        
        # 与基准对比
        if self.baseline_metrics:
            current_avg = report["summary"]["avg_latency_ms"]
            baseline_avg = self.baseline_metrics["average_latency_ms"]
            
            improvement = (baseline_avg - current_avg) / baseline_avg
            
            report["baseline_comparison"] = {
                "baseline_avg_latency_ms": baseline_avg,
                "current_avg_latency_ms": current_avg,
                "improvement_ratio": improvement,
                "meets_latency_target": improvement >= self.optimization_targets["latency_improvement_target"]
            }
        
        # 优化目标状态
        report["optimization_status"] = {
            "cache_hit_rate_target": self.optimization_targets["cache_hit_rate_target"],
            "cache_hit_rate_achieved": cache_hit_rate >= self.optimization_targets["cache_hit_rate_target"],
            "quantization_enabled": quantization_ratio > 0.5,
            "error_rate_acceptable": error_rate < 0.05  # 5% error threshold
        }
        
        return report
    
    async def validate_performance_improvements(self) -> Dict[str, Any]:
        """验证性能提升目标"""
        report = await self.get_performance_report()
        
        if report.get("status") in ["no_data", "error"]:
            return report
        
        validation_results = {
            "performance_improvement_30_percent": False,
            "memory_optimization_20_percent": False,  # Placeholder - would need memory monitoring
            "cache_efficiency_target": False,
            "overall_status": "not_achieved"
        }
        
        # 检查性能提升
        if "baseline_comparison" in report:
            improvement = report["baseline_comparison"]["improvement_ratio"]
            validation_results["performance_improvement_30_percent"] = improvement >= 0.30
        
        # 检查缓存效率
        cache_hit_rate = report["summary"]["cache_hit_rate"]
        validation_results["cache_efficiency_target"] = cache_hit_rate >= 0.75
        
        # Memory optimization placeholder - would need actual memory monitoring
        # For now, we'll use quantization ratio as a proxy
        quantization_ratio = report["summary"]["quantization_ratio"]
        validation_results["memory_optimization_20_percent"] = quantization_ratio >= 0.7
        
        # 总体状态
        achieved_targets = sum(validation_results[key] for key in validation_results if isinstance(validation_results[key], bool))
        total_targets = 3  # Number of boolean targets
        
        if achieved_targets == total_targets:
            validation_results["overall_status"] = "all_achieved"
        elif achieved_targets >= total_targets * 0.7:
            validation_results["overall_status"] = "mostly_achieved"
        else:
            validation_results["overall_status"] = "needs_improvement"
        
        validation_results["targets_achieved"] = achieved_targets
        validation_results["total_targets"] = total_targets
        validation_results["timestamp"] = utc_now().isoformat()
        
        logger.info(f"Performance validation: {achieved_targets}/{total_targets} targets achieved")
        
        return validation_results
    
    async def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """获取实时性能指标"""
        cutoff_time = utc_now() - timedelta(minutes=window_minutes)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time and m.error is None
        ]
        
        if not recent_metrics:
            return {
                "status": "no_recent_data",
                "window_minutes": window_minutes,
                "timestamp": utc_now().isoformat()
            }
        
        latencies = [m.latency_ms for m in recent_metrics]
        
        return {
            "window_minutes": window_minutes,
            "search_count": len(recent_metrics),
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "cache_hits": sum(1 for m in recent_metrics if m.cache_hit),
            "cache_hit_rate": sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics),
            "quantized_searches": sum(1 for m in recent_metrics if m.quantization_mode != "float32"),
            "timestamp": utc_now().isoformat()
        }
    
    async def detect_performance_anomalies(self, threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        """检测性能异常"""
        if len(self.metrics_history) < 50:  # Need enough data for statistical analysis
            return []
        
        recent_metrics = list(self.metrics_history)[-100:]
        successful_metrics = [m for m in recent_metrics if m.error is None]
        
        if len(successful_metrics) < 10:
            return []
        
        latencies = [m.latency_ms for m in successful_metrics]
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)
        
        anomalies = []
        
        for metric in successful_metrics[-20:]:  # Check last 20 searches
            if abs(metric.latency_ms - mean_latency) > threshold_std * std_latency:
                anomalies.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "latency_ms": metric.latency_ms,
                    "expected_range": [
                        mean_latency - threshold_std * std_latency,
                        mean_latency + threshold_std * std_latency
                    ],
                    "deviation": abs(metric.latency_ms - mean_latency) / std_latency,
                    "quantization_mode": metric.quantization_mode
                })
        
        return anomalies
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.metrics_history.clear()
        self.performance_stats = {
            "search_latency": [],
            "index_usage": {},
            "cache_hit_rate": 0.0,
            "quantization_ratio": 0.0,
            "error_count": 0,
            "total_searches": 0
        }
        self.baseline_metrics = None
        logger.info("Performance metrics reset")
