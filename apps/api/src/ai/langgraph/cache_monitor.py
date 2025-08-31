"""
缓存监控和指标收集模块
提供详细的缓存性能监控和统计功能
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .caching import NodeCache
from .cache_factory import get_node_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """缓存性能指标"""
    # 基础统计
    hit_count: int = 0
    miss_count: int = 0
    set_count: int = 0
    delete_count: int = 0
    error_count: int = 0
    
    # 延迟统计
    get_latency_total: float = 0.0  # 总获取延迟
    set_latency_total: float = 0.0  # 总设置延迟
    get_operations: int = 0  # 获取操作次数
    set_operations: int = 0  # 设置操作次数
    
    # 存储统计
    entries_current: int = 0  # 当前条目数
    memory_usage: int = 0  # 内存使用量（字节）
    
    # 时间戳
    last_reset: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_rate
    
    @property
    def avg_get_latency(self) -> float:
        """平均获取延迟（毫秒）"""
        return (self.get_latency_total * 1000 / self.get_operations) if self.get_operations > 0 else 0.0
    
    @property
    def avg_set_latency(self) -> float:
        """平均设置延迟（毫秒）"""
        return (self.set_latency_total * 1000 / self.set_operations) if self.set_operations > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）"""
        return time.time() - self.last_reset
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "set_count": self.set_count,
            "delete_count": self.delete_count,
            "error_count": self.error_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "avg_get_latency_ms": self.avg_get_latency,
            "avg_set_latency_ms": self.avg_set_latency,
            "entries_current": self.entries_current,
            "memory_usage_bytes": self.memory_usage,
            "uptime_seconds": self.uptime_seconds,
            "last_reset": datetime.fromtimestamp(self.last_reset).isoformat()
        }
    
    def reset(self):
        """重置指标"""
        self.__init__()


class CacheMonitor:
    """缓存监控器"""
    
    def __init__(self, cache: Optional[NodeCache] = None):
        self.cache = cache or get_node_cache()
        self.metrics = CacheMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._collection_interval = 30  # 30秒收集一次
        self._running = False
    
    @asynccontextmanager
    async def measure_latency(self, operation: str):
        """测量操作延迟的上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            latency = end_time - start_time
            
            if operation == "get":
                self.metrics.get_latency_total += latency
                self.metrics.get_operations += 1
            elif operation == "set":
                self.metrics.set_latency_total += latency
                self.metrics.set_operations += 1
    
    def record_hit(self):
        """记录缓存命中"""
        self.metrics.hit_count += 1
    
    def record_miss(self):
        """记录缓存未命中"""
        self.metrics.miss_count += 1
    
    def record_set(self):
        """记录缓存设置"""
        self.metrics.set_count += 1
    
    def record_delete(self):
        """记录缓存删除"""
        self.metrics.delete_count += 1
    
    def record_error(self):
        """记录缓存错误"""
        self.metrics.error_count += 1
    
    async def collect_storage_metrics(self):
        """收集存储相关指标"""
        try:
            if hasattr(self.cache, 'get_stats'):
                stats = await self.cache.get_stats()
                self.metrics.entries_current = stats.get('cache_entries', 0)
                self.metrics.memory_usage = stats.get('redis_used_memory', 0)
        except Exception as e:
            logger.error(f"收集存储指标失败: {e}")
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        # 收集最新的存储指标
        await self.collect_storage_metrics()
        
        # 获取缓存实例的统计信息
        cache_stats = {}
        try:
            if hasattr(self.cache, 'get_stats'):
                cache_stats = await self.cache.get_stats()
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
        
        # 合并监控指标和缓存统计
        detailed_stats = self.metrics.to_dict()
        detailed_stats.update({
            "cache_backend": type(self.cache).__name__,
            "cache_config": self.cache.config.__dict__ if hasattr(self.cache, 'config') else {},
            "cache_internal_stats": cache_stats
        })
        
        return detailed_stats
    
    async def start_monitoring(self):
        """启动监控任务"""
        if self._running:
            logger.warning("缓存监控已经在运行")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("缓存监控已启动")
    
    async def stop_monitoring(self):
        """停止监控任务"""
        if not self._running:
            return
        
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("缓存监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        try:
            while self._running:
                await self.collect_storage_metrics()
                
                # 记录监控日志
                stats = self.metrics.to_dict()
                logger.info(f"缓存监控: 命中率={stats['hit_rate']:.2%}, "
                          f"条目数={stats['entries_current']}, "
                          f"平均获取延迟={stats['avg_get_latency_ms']:.2f}ms")
                
                await asyncio.sleep(self._collection_interval)
        
        except asyncio.CancelledError:
            logger.info("缓存监控任务已取消")
        except Exception as e:
            logger.error(f"缓存监控任务异常: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            "hit_rate": f"{self.metrics.hit_rate:.2%}",
            "total_operations": self.metrics.hit_count + self.metrics.miss_count,
            "avg_latency_ms": f"{self.metrics.avg_get_latency:.2f}",
            "current_entries": self.metrics.entries_current,
            "error_rate": f"{self.metrics.error_count / max(1, self.metrics.get_operations + self.metrics.set_operations):.2%}",
            "uptime": f"{self.metrics.uptime_seconds:.0f}s"
        }


class CacheHealthChecker:
    """缓存健康检查器"""
    
    def __init__(self, cache: Optional[NodeCache] = None):
        self.cache = cache or get_node_cache()
    
    async def health_check(self) -> Dict[str, Any]:
        """执行缓存健康检查"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": utc_now().isoformat()
        }
        
        # 检查缓存连接
        try:
            test_key = "health_check_test"
            test_value = {"timestamp": time.time()}
            
            # 测试设置
            set_success = await self.cache.set(test_key, test_value, ttl=10)
            health_status["checks"]["set_operation"] = {
                "status": "pass" if set_success else "fail",
                "message": "缓存设置操作" + ("成功" if set_success else "失败")
            }
            
            # 测试获取
            get_result = await self.cache.get(test_key)
            get_success = get_result is not None
            health_status["checks"]["get_operation"] = {
                "status": "pass" if get_success else "fail",
                "message": "缓存获取操作" + ("成功" if get_success else "失败")
            }
            
            # 测试删除
            delete_success = await self.cache.delete(test_key)
            health_status["checks"]["delete_operation"] = {
                "status": "pass" if delete_success else "fail",
                "message": "缓存删除操作" + ("成功" if delete_success else "失败")
            }
            
            # 检查整体健康状态
            all_passed = all(check["status"] == "pass" for check in health_status["checks"].values())
            if not all_passed:
                health_status["status"] = "degraded"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["connection"] = {
                "status": "fail",
                "message": f"缓存连接失败: {str(e)}"
            }
            logger.error(f"缓存健康检查失败: {e}")
        
        return health_status
    
    async def performance_check(self) -> Dict[str, Any]:
        """执行性能检查"""
        performance_data = {
            "latency_ms": {},
            "timestamp": utc_now().isoformat()
        }
        
        try:
            # 测试设置性能
            test_data = {"test": "performance", "timestamp": time.time()}
            start_time = time.time()
            await self.cache.set("perf_test", test_data, ttl=10)
            set_latency = (time.time() - start_time) * 1000
            performance_data["latency_ms"]["set"] = round(set_latency, 2)
            
            # 测试获取性能
            start_time = time.time()
            await self.cache.get("perf_test")
            get_latency = (time.time() - start_time) * 1000
            performance_data["latency_ms"]["get"] = round(get_latency, 2)
            
            # 清理测试数据
            await self.cache.delete("perf_test")
            
            # 性能评估
            if set_latency > 100 or get_latency > 50:
                performance_data["status"] = "slow"
                performance_data["warning"] = "缓存延迟较高"
            else:
                performance_data["status"] = "fast"
            
        except Exception as e:
            performance_data["status"] = "error"
            performance_data["error"] = str(e)
            logger.error(f"缓存性能检查失败: {e}")
        
        return performance_data


# 全局监控器实例
_global_monitor: Optional[CacheMonitor] = None


def get_cache_monitor() -> CacheMonitor:
    """获取全局缓存监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CacheMonitor()
    return _global_monitor


async def start_cache_monitoring():
    """启动缓存监控"""
    monitor = get_cache_monitor()
    await monitor.start_monitoring()


async def stop_cache_monitoring():
    """停止缓存监控"""
    monitor = get_cache_monitor()
    await monitor.stop_monitoring()