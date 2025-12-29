"""
性能优化配置

针对行为分析系统的性能优化参数和配置。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os

@dataclass
class EventCollectionConfig:
    """事件收集性能配置"""
    buffer_size: int = 1000
    flush_interval: int = 30  # 秒
    batch_size: int = 100
    compression_enabled: bool = True
    compression_method: str = "gzip"  # gzip, lz4, zstd
    quality_monitoring: bool = True
    max_memory_usage_mb: int = 100

@dataclass
class DatabaseConfig:
    """数据库性能配置"""
    connection_pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    query_timeout: int = 30
    batch_insert_size: int = 1000
    enable_partitioning: bool = True
    partition_interval: str = "1 day"
    index_maintenance: bool = True

@dataclass
class AnalysisConfig:
    """分析算法性能配置"""
    max_events_per_analysis: int = 100000
    pattern_mining_min_support: float = 0.01
    clustering_max_samples: int = 10000
    anomaly_detection_window: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 500
    cache_analysis_results: bool = True
    cache_ttl_seconds: int = 3600

@dataclass
class WebSocketConfig:
    """WebSocket性能配置"""
    max_connections: int = 1000
    connection_timeout: int = 300
    heartbeat_interval: int = 30
    message_queue_size: int = 10000
    broadcast_batch_size: int = 100
    compression: bool = True

@dataclass
class CacheConfig:
    """缓存性能配置"""
    enabled: bool = True
    backend: str = "redis"  # redis, memory, memcached
    ttl_seconds: int = 3600
    max_memory_mb: int = 200
    eviction_policy: str = "lru"
    compression: bool = True
    key_prefix: str = "analytics:"

@dataclass
class PerformanceConfig:
    """综合性能配置"""
    event_collection: EventCollectionConfig
    database: DatabaseConfig
    analysis: AnalysisConfig
    websocket: WebSocketConfig
    cache: CacheConfig
    
    # 全局设置
    enable_monitoring: bool = True
    log_performance_metrics: bool = True
    profile_slow_operations: bool = True
    slow_operation_threshold_ms: int = 1000
    enable_async_processing: bool = True
    
    @classmethod
    def from_environment(cls) -> "PerformanceConfig":
        """从环境变量创建配置"""
        return cls(
            event_collection=EventCollectionConfig(
                buffer_size=int(os.getenv("ANALYTICS_BUFFER_SIZE", "1000")),
                flush_interval=int(os.getenv("ANALYTICS_FLUSH_INTERVAL", "30")),
                batch_size=int(os.getenv("ANALYTICS_BATCH_SIZE", "100")),
                compression_enabled=os.getenv("ANALYTICS_COMPRESSION", "true").lower() == "true",
                compression_method=os.getenv("ANALYTICS_COMPRESSION_METHOD", "gzip"),
                max_memory_usage_mb=int(os.getenv("ANALYTICS_MAX_MEMORY_MB", "100"))
            ),
            database=DatabaseConfig(
                connection_pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                query_timeout=int(os.getenv("DB_QUERY_TIMEOUT", "30")),
                batch_insert_size=int(os.getenv("DB_BATCH_INSERT_SIZE", "1000")),
                enable_partitioning=os.getenv("DB_ENABLE_PARTITIONING", "true").lower() == "true"
            ),
            analysis=AnalysisConfig(
                max_events_per_analysis=int(os.getenv("ANALYSIS_MAX_EVENTS", "100000")),
                max_workers=int(os.getenv("ANALYSIS_MAX_WORKERS", "4")),
                memory_limit_mb=int(os.getenv("ANALYSIS_MEMORY_LIMIT_MB", "500")),
                parallel_processing=os.getenv("ANALYSIS_PARALLEL", "true").lower() == "true",
                cache_analysis_results=os.getenv("ANALYSIS_CACHE", "true").lower() == "true"
            ),
            websocket=WebSocketConfig(
                max_connections=int(os.getenv("WS_MAX_CONNECTIONS", "1000")),
                connection_timeout=int(os.getenv("WS_CONNECTION_TIMEOUT", "300")),
                heartbeat_interval=int(os.getenv("WS_HEARTBEAT_INTERVAL", "30")),
                message_queue_size=int(os.getenv("WS_MESSAGE_QUEUE_SIZE", "10000"))
            ),
            cache=CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                backend=os.getenv("CACHE_BACKEND", "redis"),
                ttl_seconds=int(os.getenv("CACHE_TTL", "3600")),
                max_memory_mb=int(os.getenv("CACHE_MAX_MEMORY_MB", "200"))
            ),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            log_performance_metrics=os.getenv("LOG_PERFORMANCE", "true").lower() == "true",
            enable_async_processing=os.getenv("ENABLE_ASYNC", "true").lower() == "true"
        )
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """获取优化建议"""
        recommendations = {}
        
        # 事件收集优化建议
        if self.event_collection.buffer_size < 500:
            recommendations["event_buffer"] = "增加缓冲区大小以提高批处理效率"
        
        if not self.event_collection.compression_enabled:
            recommendations["compression"] = "启用压缩可以减少内存和网络使用"
        
        # 数据库优化建议
        if self.database.connection_pool_size < 10:
            recommendations["db_pool"] = "增加数据库连接池大小以提高并发性能"
        
        if not self.database.enable_partitioning:
            recommendations["partitioning"] = "启用分区可以提高大数据量查询性能"
        
        # 分析优化建议
        if not self.analysis.parallel_processing:
            recommendations["parallel"] = "启用并行处理可以显著提高分析速度"
        
        if not self.analysis.cache_analysis_results:
            recommendations["cache"] = "启用结果缓存可以避免重复计算"
        
        # 缓存优化建议
        if not self.cache.enabled:
            recommendations["enable_cache"] = "启用缓存可以显著提高响应速度"
        
        return recommendations
    
    def validate_configuration(self) -> Dict[str, Any]:
        """验证配置有效性"""
        issues = []
        warnings = []
        
        # 验证缓冲区大小
        if self.event_collection.buffer_size > 10000:
            warnings.append("缓冲区大小过大可能导致内存问题")
        
        # 验证数据库配置
        if self.database.connection_pool_size > 100:
            warnings.append("数据库连接池过大可能影响数据库性能")
        
        # 验证分析配置
        if self.analysis.max_events_per_analysis > 1000000:
            issues.append("单次分析事件数量过大，可能导致内存溢出")
        
        # 验证WebSocket配置
        if self.websocket.max_connections > 10000:
            warnings.append("WebSocket最大连接数过大，确保服务器资源充足")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics = {}
    
    def optimize_for_workload(self, workload_characteristics: Dict[str, Any]) -> PerformanceConfig:
        """根据工作负载特征优化配置"""
        optimized_config = self.config
        
        # 高频事件收集优化
        if workload_characteristics.get("high_frequency_events", False):
            optimized_config.event_collection.buffer_size = min(5000, optimized_config.event_collection.buffer_size * 2)
            optimized_config.event_collection.flush_interval = max(10, optimized_config.event_collection.flush_interval // 2)
        
        # 大数据量分析优化
        if workload_characteristics.get("large_dataset_analysis", False):
            optimized_config.analysis.parallel_processing = True
            optimized_config.analysis.max_workers = min(8, optimized_config.analysis.max_workers * 2)
            optimized_config.database.batch_insert_size = min(2000, optimized_config.database.batch_insert_size * 2)
        
        # 实时性要求优化
        if workload_characteristics.get("real_time_requirements", False):
            optimized_config.event_collection.flush_interval = min(10, optimized_config.event_collection.flush_interval)
            optimized_config.websocket.heartbeat_interval = min(15, optimized_config.websocket.heartbeat_interval)
            optimized_config.cache.ttl_seconds = min(1800, optimized_config.cache.ttl_seconds)
        
        # 内存限制优化
        if workload_characteristics.get("memory_constrained", False):
            optimized_config.event_collection.buffer_size = max(100, optimized_config.event_collection.buffer_size // 2)
            optimized_config.analysis.max_events_per_analysis = max(10000, optimized_config.analysis.max_events_per_analysis // 2)
            optimized_config.cache.max_memory_mb = max(50, optimized_config.cache.max_memory_mb // 2)
        
        return optimized_config
    
    def analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """分析性能瓶颈"""
        bottlenecks = {}
        
        # 分析缓冲区性能
        if metrics.get("buffer_overflow_rate", 0) > 0.1:
            bottlenecks["buffer"] = "缓冲区溢出率过高，建议增加缓冲区大小或减少刷新间隔"
        
        # 分析数据库性能
        if metrics.get("db_query_avg_time", 0) > 1000:  # 1秒
            bottlenecks["database"] = "数据库查询时间过长，建议优化查询或增加索引"
        
        # 分析内存使用
        if metrics.get("memory_usage_percent", 0) > 80:
            bottlenecks["memory"] = "内存使用率过高，建议增加内存限制或优化数据结构"
        
        # 分析WebSocket性能
        if metrics.get("ws_message_drop_rate", 0) > 0.05:
            bottlenecks["websocket"] = "WebSocket消息丢失率过高，建议增加队列大小或优化广播策略"
        
        return bottlenecks
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        validation = self.config.validate_configuration()
        recommendations = self.config.get_optimization_recommendations()
        
        return {
            "configuration_status": validation,
            "optimization_recommendations": recommendations,
            "current_settings": {
                "event_collection": {
                    "buffer_size": self.config.event_collection.buffer_size,
                    "flush_interval": self.config.event_collection.flush_interval,
                    "compression": self.config.event_collection.compression_enabled
                },
                "database": {
                    "pool_size": self.config.database.connection_pool_size,
                    "partitioning": self.config.database.enable_partitioning
                },
                "analysis": {
                    "parallel_processing": self.config.analysis.parallel_processing,
                    "max_workers": self.config.analysis.max_workers,
                    "caching": self.config.analysis.cache_analysis_results
                }
            }
        }

# 全局配置实例
performance_config = PerformanceConfig.from_environment()
performance_optimizer = PerformanceOptimizer(performance_config)
