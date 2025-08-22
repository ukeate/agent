"""
API性能指标收集
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Request, Response
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.redis import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class APIMetrics(BaseModel):
    """API指标模型"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_id: Optional[str]
    api_key: Optional[str]
    timestamp: datetime
    cache_hit: bool = False
    cache_key: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.redis = None
        self.metrics_buffer: List[APIMetrics] = []
        self.buffer_size = 100
        self.flush_interval = 60  # 秒
        self.enabled = settings.VECTOR_MONITORING_ENABLED
        
        # 实时统计
        self.endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "min_time": float("inf"),
            "max_time": 0,
            "errors": 0,
            "cache_hits": 0
        })
        
        # 慢查询记录
        self.slow_queries: List[APIMetrics] = []
        self.slow_query_threshold = settings.VECTOR_SLOW_QUERY_THRESHOLD * 1000  # 转换为毫秒
    
    async def initialize(self):
        """初始化指标收集器"""
        if self.enabled:
            try:
                self.redis = await get_redis()
                logger.info("Metrics collector initialized")
            except Exception as e:
                logger.error("Failed to initialize metrics collector", error=str(e))
                self.enabled = False
    
    async def collect(
        self,
        request: Request,
        response: Response,
        start_time: float,
        error: Optional[Exception] = None
    ):
        """收集API指标"""
        if not self.enabled:
            return
        
        # 计算响应时间
        response_time_ms = (time.time() - start_time) * 1000
        
        # 获取请求和响应大小
        request_size = int(request.headers.get("content-length", 0))
        response_size = int(response.headers.get("content-length", 0))
        
        # 获取用户信息
        user_id = None
        api_key = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)
        if hasattr(request.state, "api_key"):
            api_key = request.state.api_key
        
        # 检查缓存命中
        cache_hit = response.headers.get("X-Cache") == "HIT"
        cache_key = response.headers.get("X-Cache-Key")
        
        # 创建指标对象
        metric = APIMetrics(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            user_id=user_id,
            api_key=api_key,
            timestamp=datetime.utcnow(),
            cache_hit=cache_hit,
            cache_key=cache_key,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None
        )
        
        # 添加到缓冲区
        self.metrics_buffer.append(metric)
        
        # 更新实时统计
        self._update_stats(metric)
        
        # 检查慢查询
        if response_time_ms > self.slow_query_threshold:
            self.slow_queries.append(metric)
            logger.warning(
                "Slow API call detected",
                endpoint=metric.endpoint,
                response_time_ms=response_time_ms,
                threshold_ms=self.slow_query_threshold
            )
        
        # 如果缓冲区满了，刷新到存储
        if len(self.metrics_buffer) >= self.buffer_size:
            await self.flush_buffer()
    
    def _update_stats(self, metric: APIMetrics):
        """更新实时统计"""
        key = f"{metric.endpoint}:{metric.method}"
        stats = self.endpoint_stats[key]
        
        stats["count"] += 1
        stats["total_time"] += metric.response_time_ms
        stats["min_time"] = min(stats["min_time"], metric.response_time_ms)
        stats["max_time"] = max(stats["max_time"], metric.response_time_ms)
        
        if metric.status_code >= 400:
            stats["errors"] += 1
        
        if metric.cache_hit:
            stats["cache_hits"] += 1
    
    async def flush_buffer(self):
        """刷新缓冲区到存储"""
        if not self.metrics_buffer or not self.redis:
            return
        
        try:
            # 批量写入Redis
            pipeline = self.redis.pipeline()
            
            for metric in self.metrics_buffer:
                # 存储到时间序列
                key = f"metrics:api:{metric.endpoint}:{metric.method}"
                score = metric.timestamp.timestamp()
                value = metric.json()
                
                # 添加到有序集合（保留1天的数据）
                pipeline.zadd(key, {value: score})
                pipeline.expire(key, 86400)
                
                # 更新聚合统计
                stats_key = f"metrics:stats:{metric.endpoint}:{metric.method}"
                pipeline.hincrby(stats_key, "count", 1)
                pipeline.hincrbyfloat(stats_key, "total_time", metric.response_time_ms)
                
                if metric.status_code >= 400:
                    pipeline.hincrby(stats_key, "errors", 1)
                
                if metric.cache_hit:
                    pipeline.hincrby(stats_key, "cache_hits", 1)
                
                pipeline.expire(stats_key, 86400)
            
            await pipeline.execute()
            
            # 清空缓冲区
            self.metrics_buffer.clear()
            
            logger.debug("Metrics buffer flushed", count=len(self.metrics_buffer))
            
        except Exception as e:
            logger.error("Failed to flush metrics buffer", error=str(e))
    
    async def get_endpoint_metrics(
        self,
        endpoint: str,
        method: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取端点指标"""
        if not self.redis:
            return {}
        
        key = f"metrics:api:{endpoint}:{method}"
        
        # 设置时间范围
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        start_score = start_time.timestamp()
        end_score = end_time.timestamp()
        
        try:
            # 从Redis获取指标
            results = await self.redis.zrangebyscore(
                key,
                start_score,
                end_score,
                withscores=False
            )
            
            metrics = [APIMetrics.parse_raw(r) for r in results]
            
            # 计算统计
            if metrics:
                response_times = [m.response_time_ms for m in metrics]
                return {
                    "endpoint": endpoint,
                    "method": method,
                    "count": len(metrics),
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "p50_response_time": self._percentile(response_times, 50),
                    "p95_response_time": self._percentile(response_times, 95),
                    "p99_response_time": self._percentile(response_times, 99),
                    "error_rate": sum(1 for m in metrics if m.status_code >= 400) / len(metrics),
                    "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics)
                }
            
        except Exception as e:
            logger.error("Failed to get endpoint metrics", error=str(e))
        
        return {}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时统计"""
        stats = {}
        
        for key, data in self.endpoint_stats.items():
            if data["count"] > 0:
                stats[key] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "min_time": data["min_time"],
                    "max_time": data["max_time"],
                    "error_rate": data["errors"] / data["count"],
                    "cache_hit_rate": data["cache_hits"] / data["count"]
                }
        
        return stats
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取慢查询"""
        # 按响应时间排序
        sorted_queries = sorted(
            self.slow_queries,
            key=lambda x: x.response_time_ms,
            reverse=True
        )
        
        return [q.dict() for q in sorted_queries[:limit]]
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        real_time_stats = self.get_real_time_stats()
        slow_queries = self.get_slow_queries()
        
        # 计算总体指标
        total_requests = sum(s["count"] for s in real_time_stats.values())
        
        if total_requests > 0:
            avg_response_time = sum(
                s["avg_time"] * s["count"]
                for s in real_time_stats.values()
            ) / total_requests
            
            total_errors = sum(
                s["count"] * s["error_rate"]
                for s in real_time_stats.values()
            )
            
            total_cache_hits = sum(
                s["count"] * s["cache_hit_rate"]
                for s in real_time_stats.values()
            )
        else:
            avg_response_time = 0
            total_errors = 0
            total_cache_hits = 0
        
        return {
            "total_requests": total_requests,
            "avg_response_time_ms": avg_response_time,
            "total_errors": int(total_errors),
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "cache_hit_rate": total_cache_hits / total_requests if total_requests > 0 else 0,
            "slow_queries_count": len(self.slow_queries),
            "top_slow_queries": slow_queries[:5],
            "endpoint_stats": real_time_stats
        }


# 全局指标收集器实例
metrics_collector = MetricsCollector()