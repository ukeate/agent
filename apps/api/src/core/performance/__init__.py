"""
性能优化模块
"""

from src.core.performance.caching import (
    CacheKey,
    CacheManager,
    CachedResponse,
    cache_invalidate,
    cache_manager,
    cache_response,
)
from src.core.performance.compression import CompressionHandler, compression_handler
from src.core.performance.connection_pool import ConnectionPoolManager, pool_manager
from src.core.performance.metrics import (
    APIMetrics,
    MetricsCollector,
    metrics_collector,
)

__all__ = [
    # Caching
    "CacheKey",
    "CacheManager",
    "CachedResponse",
    "cache_invalidate",
    "cache_manager",
    "cache_response",
    # Compression
    "CompressionHandler",
    "compression_handler",
    # Connection Pool
    "ConnectionPoolManager",
    "pool_manager",
    # Metrics
    "APIMetrics",
    "MetricsCollector",
    "metrics_collector",
]