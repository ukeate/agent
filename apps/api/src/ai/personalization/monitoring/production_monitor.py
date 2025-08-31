"""
生产环境监控系统
集成告警、指标收集、健康检查等功能，确保个性化引擎生产就绪
"""

import asyncio
import logging
import time
import psutil
import gc
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

import aioredis
import numpy as np
from contextlib import asynccontextmanager

from .alerts import AlertManager, AlertSeverity, get_default_alert_rules
from .performance import PerformanceMonitor
from ..engine import PersonalizationEngine


logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float
    details: Dict[str, Any]
    timestamp: datetime


class ProductionMonitor:
    """生产环境监控器"""
    
    def __init__(
        self,
        personalization_engine: PersonalizationEngine,
        redis_client,
        alert_manager: AlertManager,
        performance_monitor: PerformanceMonitor
    ):
        self.engine = personalization_engine
        self.redis = redis_client
        self.alert_manager = alert_manager
        self.performance_monitor = performance_monitor
        
        # 监控状态
        self._running = False
        self._tasks = []
        self._last_metrics_update = utc_now()
        
        # 监控配置
        self.config = {
            "metrics_interval": 30,          # 指标收集间隔（秒）
            "health_check_interval": 60,     # 健康检查间隔（秒）
            "cleanup_interval": 300,         # 清理间隔（秒）
            "retention_hours": 24,           # 数据保留时间（小时）
        }
        
        # 性能基线
        self.baselines = {
            "recommendation_latency_p99": 100.0,
            "feature_computation_latency_avg": 10.0,
            "cache_hit_rate": 0.8,
            "error_rate": 0.01,
            "memory_usage_percent": 80.0,
            "cpu_usage_percent": 70.0
        }
    
    async def start(self):
        """启动生产监控"""
        if self._running:
            return
        
        self._running = True
        logger.info("启动生产环境监控")
        
        # 启动告警管理器
        await self.alert_manager.start()
        
        # 添加默认告警规则
        for rule in get_default_alert_rules():
            self.alert_manager.add_rule(rule)
        
        # 启动监控任务
        self._tasks = [
            asyncio.create_task(self._collect_metrics_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._performance_analysis_loop())
        ]
        
        logger.info("生产监控已启动")
    
    async def stop(self):
        """停止生产监控"""
        if not self._running:
            return
        
        self._running = False
        logger.info("停止生产环境监控")
        
        # 停止所有任务
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # 停止告警管理器
        await self.alert_manager.stop()
        
        logger.info("生产监控已停止")
    
    async def _collect_metrics_loop(self):
        """指标收集循环"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_business_metrics()
                
                self._last_metrics_update = utc_now()
                await asyncio.sleep(self.config["metrics_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
                await asyncio.sleep(10)  # 短暂等待后重试
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.alert_manager.update_metric("cpu_usage_percent", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            await self.alert_manager.update_metric("memory_usage_percent", memory_percent)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.alert_manager.update_metric("disk_usage_percent", disk_percent)
            
            # 网络I/O
            network = psutil.net_io_counters()
            await self.alert_manager.update_metric("network_bytes_sent", network.bytes_sent)
            await self.alert_manager.update_metric("network_bytes_recv", network.bytes_recv)
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            await self.alert_manager.update_metric("process_memory_mb", process_memory)
            
            # 文件描述符
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            await self.alert_manager.update_metric("open_file_descriptors", num_fds)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    async def _collect_application_metrics(self):
        """收集应用指标"""
        try:
            # 从性能监控器获取指标
            metrics = await self.performance_monitor.get_metrics()
            
            for metric_name, value in metrics.items():
                await self.alert_manager.update_metric(metric_name, value)
            
            # Redis连接池状态
            if hasattr(self.redis, 'connection_pool'):
                pool = self.redis.connection_pool
                await self.alert_manager.update_metric(
                    "redis_connections_created", 
                    getattr(pool, 'created_connections', 0)
                )
                await self.alert_manager.update_metric(
                    "redis_connections_available", 
                    getattr(pool, 'available_connections', 0)
                )
            
            # 垃圾回收统计
            gc_stats = gc.get_stats()
            if gc_stats:
                await self.alert_manager.update_metric("gc_collections", sum(stat['collections'] for stat in gc_stats))
                await self.alert_manager.update_metric("gc_collected", sum(stat['collected'] for stat in gc_stats))
            
        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")
    
    async def _collect_business_metrics(self):
        """收集业务指标"""
        try:
            # 推荐请求统计
            total_requests = await self.redis.get("stats:total_recommendations") or 0
            await self.alert_manager.update_metric("total_recommendations", float(total_requests))
            
            # 用户活跃度
            active_users = await self.redis.scard("stats:active_users") or 0
            await self.alert_manager.update_metric("active_users", active_users)
            
            # 缓存统计
            cache_stats = await self._get_cache_statistics()
            for metric_name, value in cache_stats.items():
                await self.alert_manager.update_metric(metric_name, value)
            
            # 模型统计
            model_stats = await self._get_model_statistics()
            for metric_name, value in model_stats.items():
                await self.alert_manager.update_metric(metric_name, value)
            
        except Exception as e:
            logger.error(f"收集业务指标失败: {e}")
    
    async def _get_cache_statistics(self) -> Dict[str, float]:
        """获取缓存统计"""
        try:
            # 计算缓存命中率
            cache_hits = float(await self.redis.get("stats:cache_hits") or 0)
            cache_misses = float(await self.redis.get("stats:cache_misses") or 0)
            total_cache_requests = cache_hits + cache_misses
            
            if total_cache_requests > 0:
                cache_hit_rate = cache_hits / total_cache_requests
            else:
                cache_hit_rate = 0.0
            
            return {
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "total_cache_requests": total_cache_requests
            }
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}
    
    async def _get_model_statistics(self) -> Dict[str, float]:
        """获取模型统计"""
        try:
            # 模型预测统计
            successful_predictions = float(await self.redis.get("stats:model_predictions_success") or 0)
            failed_predictions = float(await self.redis.get("stats:model_predictions_failed") or 0)
            total_predictions = successful_predictions + failed_predictions
            
            if total_predictions > 0:
                prediction_failure_rate = failed_predictions / total_predictions
            else:
                prediction_failure_rate = 0.0
            
            return {
                "model_prediction_failure_rate": prediction_failure_rate,
                "model_predictions_success": successful_predictions,
                "model_predictions_failed": failed_predictions,
                "total_model_predictions": total_predictions
            }
            
        except Exception as e:
            logger.error(f"获取模型统计失败: {e}")
            return {}
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                health_results = await self._perform_health_checks()
                await self._process_health_results(health_results)
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_checks(self) -> List[HealthCheckResult]:
        """执行健康检查"""
        health_checks = [
            self._check_redis_health(),
            self._check_personalization_engine_health(),
            self._check_feature_engine_health(),
            self._check_model_service_health(),
            self._check_cache_health()
        ]
        
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, HealthCheckResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"健康检查异常: {result}")
        
        return valid_results
    
    async def _check_redis_health(self) -> HealthCheckResult:
        """检查Redis健康状态"""
        start_time = time.time()
        try:
            # Ping Redis
            await self.redis.ping()
            
            # 检查基本操作
            test_key = "health_check_test"
            await self.redis.setex(test_key, 10, "test_value")
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            latency_ms = (time.time() - start_time) * 1000
            
            status = "healthy" if latency_ms < 50 else "degraded"
            
            return HealthCheckResult(
                service="redis",
                status=status,
                latency_ms=latency_ms,
                details={
                    "ping_success": True,
                    "read_write_success": value == "test_value"
                },
                timestamp=utc_now()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="redis",
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)},
                timestamp=utc_now()
            )
    
    async def _check_personalization_engine_health(self) -> HealthCheckResult:
        """检查个性化引擎健康状态"""
        start_time = time.time()
        try:
            # 模拟推荐请求
            from models.schemas.personalization import RecommendationRequest
            
            test_request = RecommendationRequest(
                user_id="health_check_user",
                context={"health_check": True},
                n_recommendations=3,
                scenario="content_discovery"
            )
            
            response = await self.engine.get_recommendations(test_request)
            latency_ms = (time.time() - start_time) * 1000
            
            status = "healthy" if latency_ms < 200 else "degraded"
            
            return HealthCheckResult(
                service="personalization_engine",
                status=status,
                latency_ms=latency_ms,
                details={
                    "recommendations_count": len(response.recommendations),
                    "response_complete": response.request_id is not None
                },
                timestamp=utc_now()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="personalization_engine",
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)},
                timestamp=utc_now()
            )
    
    async def _check_feature_engine_health(self) -> HealthCheckResult:
        """检查特征引擎健康状态"""
        start_time = time.time()
        try:
            # 测试特征计算
            features = await self.engine.feature_engine.compute_features(
                "health_check_user",
                {"health_check": True}
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            status = "healthy" if latency_ms < 50 else "degraded"
            
            return HealthCheckResult(
                service="feature_engine",
                status=status,
                latency_ms=latency_ms,
                details={
                    "features_computed": len(features) if features else 0,
                    "has_temporal": "temporal" in features if features else False,
                    "has_behavioral": "behavioral" in features if features else False
                },
                timestamp=utc_now()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="feature_engine",
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)},
                timestamp=utc_now()
            )
    
    async def _check_model_service_health(self) -> HealthCheckResult:
        """检查模型服务健康状态"""
        start_time = time.time()
        try:
            # 测试模型预测
            test_features = np.random.rand(10).astype(np.float32)
            result = await self.engine.model_service.predict(test_features)
            
            latency_ms = (time.time() - start_time) * 1000
            
            status = "healthy" if latency_ms < 100 else "degraded"
            
            return HealthCheckResult(
                service="model_service",
                status=status,
                latency_ms=latency_ms,
                details={
                    "prediction_shape": result.shape if hasattr(result, 'shape') else None,
                    "prediction_success": result is not None
                },
                timestamp=utc_now()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="model_service",
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)},
                timestamp=utc_now()
            )
    
    async def _check_cache_health(self) -> HealthCheckResult:
        """检查缓存健康状态"""
        start_time = time.time()
        try:
            # 测试缓存操作
            cache_key = "health_check_cache"
            test_data = {"test": "health_check", "timestamp": time.time()}
            
            # 写入缓存
            await self.engine.result_cache.set(cache_key, test_data, ttl=30)
            
            # 读取缓存
            cached_data = await self.engine.result_cache.get(cache_key)
            
            # 清理
            await self.engine.result_cache.delete(cache_key)
            
            latency_ms = (time.time() - start_time) * 1000
            
            cache_working = cached_data is not None and cached_data.get("test") == "health_check"
            status = "healthy" if cache_working and latency_ms < 20 else "degraded"
            
            return HealthCheckResult(
                service="cache",
                status=status,
                latency_ms=latency_ms,
                details={
                    "cache_read_success": cached_data is not None,
                    "cache_data_correct": cache_working
                },
                timestamp=utc_now()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="cache",
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)},
                timestamp=utc_now()
            )
    
    async def _process_health_results(self, health_results: List[HealthCheckResult]):
        """处理健康检查结果"""
        for result in health_results:
            # 保存健康检查结果
            await self.redis.hset(
                f"health_check:{result.service}",
                mapping={
                    "status": result.status,
                    "latency_ms": result.latency_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "details": json.dumps(result.details)
                }
            )
            
            # 设置过期时间
            await self.redis.expire(f"health_check:{result.service}", 3600)
            
            # 更新延迟指标
            await self.alert_manager.update_metric(
                f"{result.service}_health_check_latency",
                result.latency_ms
            )
            
            # 如果服务不健康，触发告警
            if result.status == "unhealthy":
                await self.alert_manager.update_metric(
                    f"{result.service}_health_status",
                    0.0  # 0表示不健康
                )
            else:
                await self.alert_manager.update_metric(
                    f"{result.service}_health_status",
                    1.0  # 1表示健康
                )
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.config["cleanup_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"数据清理失败: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            cutoff_time = utc_now() - timedelta(hours=self.config["retention_hours"])
            cutoff_timestamp = cutoff_time.timestamp()
            
            # 清理旧的指标数据
            metric_keys = await self.redis.keys("metric:*")
            for key in metric_keys:
                await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)
            
            # 清理旧的日志数据
            log_keys = await self.redis.keys("*_log:*")
            for key in log_keys:
                # 检查键的创建时间（如果可能的话）
                ttl = await self.redis.ttl(key)
                if ttl == -1:  # 没有设置过期时间
                    await self.redis.expire(key, 86400)  # 设置24小时过期
            
            logger.debug("已清理旧数据")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    async def _system_monitoring_loop(self):
        """系统监控循环"""
        while self._running:
            try:
                await self._monitor_system_health()
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"系统监控失败: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_system_health(self):
        """监控系统健康状态"""
        try:
            # 检查内存泄漏
            current_memory = psutil.virtual_memory().percent
            if current_memory > 90:
                logger.warning(f"内存使用率过高: {current_memory}%")
            
            # 检查文件描述符泄漏
            process = psutil.Process()
            if hasattr(process, 'num_fds'):
                num_fds = process.num_fds()
                if num_fds > 1000:
                    logger.warning(f"文件描述符数量过多: {num_fds}")
            
            # 检查网络连接
            connections = process.connections()
            if len(connections) > 500:
                logger.warning(f"网络连接数过多: {len(connections)}")
            
            # 检查线程数
            num_threads = process.num_threads()
            if num_threads > 100:
                logger.warning(f"线程数过多: {num_threads}")
            
        except Exception as e:
            logger.error(f"系统健康监控失败: {e}")
    
    async def _performance_analysis_loop(self):
        """性能分析循环"""
        while self._running:
            try:
                await self._analyze_performance_trends()
                await asyncio.sleep(300)  # 每5分钟分析一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能分析失败: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_performance_trends(self):
        """分析性能趋势"""
        try:
            # 分析延迟趋势
            await self._analyze_latency_trends()
            
            # 分析吞吐量趋势
            await self._analyze_throughput_trends()
            
            # 分析错误率趋势
            await self._analyze_error_rate_trends()
            
        except Exception as e:
            logger.error(f"性能趋势分析失败: {e}")
    
    async def _analyze_latency_trends(self):
        """分析延迟趋势"""
        try:
            # 获取最近的延迟数据
            latency_data = await self.redis.zrange(
                "metric:recommendation_latency_p99",
                -60,  # 最近60个数据点
                -1,
                withscores=True
            )
            
            if len(latency_data) >= 10:
                values = [float(score) for value, score in latency_data]
                
                # 计算趋势
                recent_avg = np.mean(values[-10:])
                previous_avg = np.mean(values[-20:-10]) if len(values) >= 20 else recent_avg
                
                trend_ratio = recent_avg / previous_avg if previous_avg > 0 else 1.0
                
                # 如果延迟增长超过20%，记录警告
                if trend_ratio > 1.2:
                    logger.warning(f"延迟趋势恶化: 近期平均 {recent_avg:.2f}ms, 之前平均 {previous_avg:.2f}ms")
                    
                    # 更新趋势指标
                    await self.alert_manager.update_metric("latency_trend_ratio", trend_ratio)
            
        except Exception as e:
            logger.error(f"延迟趋势分析失败: {e}")
    
    async def _analyze_throughput_trends(self):
        """分析吞吐量趋势"""
        try:
            # 获取QPS数据
            qps_data = await self.redis.zrange(
                "metric:requests_per_second",
                -60,
                -1,
                withscores=True
            )
            
            if len(qps_data) >= 10:
                values = [float(score) for value, score in qps_data]
                
                recent_avg = np.mean(values[-10:])
                previous_avg = np.mean(values[-20:-10]) if len(values) >= 20 else recent_avg
                
                trend_ratio = recent_avg / previous_avg if previous_avg > 0 else 1.0
                
                # 如果吞吐量下降超过20%，记录警告
                if trend_ratio < 0.8:
                    logger.warning(f"吞吐量趋势下降: 近期平均 {recent_avg:.2f} QPS, 之前平均 {previous_avg:.2f} QPS")
                    
                    await self.alert_manager.update_metric("throughput_trend_ratio", trend_ratio)
            
        except Exception as e:
            logger.error(f"吞吐量趋势分析失败: {e}")
    
    async def _analyze_error_rate_trends(self):
        """分析错误率趋势"""
        try:
            # 获取错误率数据
            error_data = await self.redis.zrange(
                "metric:error_rate",
                -60,
                -1,
                withscores=True
            )
            
            if len(error_data) >= 10:
                values = [float(score) for value, score in error_data]
                
                recent_avg = np.mean(values[-10:])
                previous_avg = np.mean(values[-20:-10]) if len(values) >= 20 else recent_avg
                
                # 如果错误率显著增加，记录警告
                if recent_avg > previous_avg * 1.5 and recent_avg > 0.005:  # 增长50%且超过0.5%
                    logger.warning(f"错误率趋势恶化: 近期平均 {recent_avg:.4f}, 之前平均 {previous_avg:.4f}")
                    
                    await self.alert_manager.update_metric("error_rate_trend", recent_avg / previous_avg if previous_avg > 0 else 1.0)
            
        except Exception as e:
            logger.error(f"错误率趋势分析失败: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 获取健康检查结果
            health_services = ["redis", "personalization_engine", "feature_engine", "model_service", "cache"]
            health_status = {}
            
            for service in health_services:
                health_data = await self.redis.hgetall(f"health_check:{service}")
                if health_data:
                    health_status[service] = {
                        "status": health_data.get("status", "unknown"),
                        "latency_ms": float(health_data.get("latency_ms", 0)),
                        "last_check": health_data.get("timestamp", "")
                    }
                else:
                    health_status[service] = {
                        "status": "unknown",
                        "latency_ms": 0,
                        "last_check": ""
                    }
            
            # 获取活跃告警
            active_alerts = await self.alert_manager.get_active_alerts()
            
            # 获取关键指标
            key_metrics = {}
            metric_names = [
                "recommendation_latency_p99",
                "feature_computation_latency_avg",
                "cache_hit_rate",
                "error_rate",
                "requests_per_second"
            ]
            
            for metric_name in metric_names:
                latest_data = await self.redis.zrange(f"metric:{metric_name}", -1, -1, withscores=True)
                if latest_data:
                    key_metrics[metric_name] = float(latest_data[0][1])
                else:
                    key_metrics[metric_name] = 0.0
            
            # 系统资源
            system_resources = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
            
            return {
                "timestamp": utc_now().isoformat(),
                "overall_status": self._calculate_overall_status(health_status, active_alerts),
                "health_checks": health_status,
                "active_alerts": [alert.dict() for alert in active_alerts],
                "key_metrics": key_metrics,
                "system_resources": system_resources,
                "uptime_seconds": (utc_now() - self._last_metrics_update).total_seconds() if self._running else 0
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                "timestamp": utc_now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    def _calculate_overall_status(self, health_status: Dict, active_alerts: List) -> str:
        """计算整体状态"""
        # 检查是否有严重告警
        critical_alerts = [alert for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return "critical"
        
        # 检查健康状态
        unhealthy_services = [service for service, status in health_status.items() if status["status"] == "unhealthy"]
        if unhealthy_services:
            return "unhealthy"
        
        # 检查是否有高级别告警
        high_alerts = [alert for alert in active_alerts if alert.severity == AlertSeverity.HIGH]
        if high_alerts:
            return "degraded"
        
        # 检查是否有降级服务
        degraded_services = [service for service, status in health_status.items() if status["status"] == "degraded"]
        if degraded_services:
            return "degraded"
        
        return "healthy"


@asynccontextmanager
async def production_monitor_context(
    personalization_engine: PersonalizationEngine,
    redis_client,
    alert_channels: List,
    performance_monitor: PerformanceMonitor
):
    """生产监控上下文管理器"""
    from .alerts import AlertManager
    
    # 创建告警管理器
    alert_manager = AlertManager(redis_client, alert_channels)
    
    # 创建生产监控器
    monitor = ProductionMonitor(
        personalization_engine,
        redis_client,
        alert_manager,
        performance_monitor
    )
    
    try:
        await monitor.start()
        yield monitor
    finally:
        await monitor.stop()