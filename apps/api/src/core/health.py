"""系统健康检查实现"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import asyncio
import psutil
import time
from fastapi import HTTPException

from src.core.database import get_db
from src.core.redis import get_redis
from src.core.logging import get_logger
from src.core.utils.timezone_utils import utc_now

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """组件健康状态"""
    def __init__(self, name: str, status: HealthStatus, details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status
        self.details = details or {}
        self.checked_at = utc_now()


class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.component_checks = []
        self.last_check_time = None
        self.check_results = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register_check("database", self._check_database)
        self.register_check("redis", self._check_redis)
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("api", self._check_api_health)
    
    def register_check(self, name: str, check_func):
        """注册健康检查函数"""
        self.component_checks.append({
            "name": name,
            "check": check_func
        })
    
    async def check_health(self, detailed: bool = False) -> Dict[str, Any]:
        """执行健康检查"""
        start_time = time.time()
        results = {}
        overall_status = HealthStatus.HEALTHY
        failed_components = []
        degraded_components = []
        
        # 并发执行所有健康检查
        check_tasks = []
        for component in self.component_checks:
            check_tasks.append(self._execute_check(component))
        
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # 处理检查结果
        for component, result in zip(self.component_checks, check_results):
            component_name = component["name"]
            
            if isinstance(result, Exception):
                # 检查失败
                results[component_name] = {
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(result)
                }
                overall_status = HealthStatus.UNHEALTHY
                failed_components.append(component_name)
            else:
                results[component_name] = result
                
                if result["status"] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                    failed_components.append(component_name)
                elif result["status"] == HealthStatus.DEGRADED:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                    degraded_components.append(component_name)
        
        # 记录检查时间
        check_duration = time.time() - start_time
        self.last_check_time = utc_now()
        self.check_results = results
        
        # 构建响应
        response = {
            "status": overall_status,
            "timestamp": self.last_check_time.isoformat(),
            "check_duration_ms": round(check_duration * 1000, 2),
            "components": results
        }
        
        if failed_components:
            response["failed_components"] = failed_components
        
        if degraded_components:
            response["degraded_components"] = degraded_components
        
        if detailed:
            response["metrics"] = await self._collect_metrics()
        
        return response
    
    async def _execute_check(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个组件检查"""
        try:
            # 设置超时
            result = await asyncio.wait_for(
                component["check"](),
                timeout=5.0  # 5秒超时
            )
            return result
        except asyncio.TimeoutError:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": "Health check timeout"
            }
        except Exception as e:
            logger.error(f"Health check failed for {component['name']}: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """检查数据库健康状态"""
        try:
            start = time.time()
            
            # 执行简单查询
            db_gen = get_db()
            db = await anext(db_gen)
            try:
                result = await db.execute("SELECT 1")
                await result.fetchone()
            finally:
                # 确保生成器被正确关闭
                try:
                    await anext(db_gen)
                except StopAsyncIteration:
                    pass
            
            response_time = (time.time() - start) * 1000  # 转换为毫秒
            
            # 根据响应时间判断状态
            if response_time > 100:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status,
                "response_time_ms": round(response_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """检查Redis健康状态"""
        try:
            redis = get_redis()
            start = time.time()
            
            # Ping Redis
            await redis.ping()
            
            response_time = (time.time() - start) * 1000
            
            # 获取Redis信息
            info = await redis.info()
            
            # 检查内存使用
            used_memory_mb = info.get("used_memory", 0) / 1024 / 1024
            
            if response_time > 50:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status,
                "response_time_ms": round(response_time, 2),
                "used_memory_mb": round(used_memory_mb, 2)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            
            # 确定健康状态
            status = HealthStatus.HEALTHY
            warnings = []
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                warnings.append("CPU usage critical")
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                warnings.append("CPU usage high")
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                warnings.append("Memory usage critical")
            elif memory.percent > 70:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append("Memory usage high")
            
            if disk.percent > 90:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append("Disk usage high")
            
            result = {
                "status": status,
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_available_mb": round(memory.available / 1024 / 1024, 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
            }
            
            if warnings:
                result["warnings"] = warnings
            
            return result
            
        except Exception as e:
            logger.error(f"System resources check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """检查API健康状态"""
        try:
            # 这里可以检查API特定的指标
            # 例如：活动连接数、请求队列长度等
            
            # 获取进程信息
            process = psutil.Process()
            connections = len(process.connections())
            threads = process.num_threads()
            
            status = HealthStatus.HEALTHY
            
            if connections > 1000:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status,
                "active_connections": connections,
                "thread_count": threads,
                "uptime_seconds": round(time.time() - process.create_time())
            }
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            process = psutil.Process()
            
            # 收集各种指标
            metrics = {
                "process": {
                    "pid": process.pid,
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, "num_fds") else None
                },
                "system": {
                    "cpu_count": psutil.cpu_count(),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {"error": str(e)}


# 全局健康检查器实例
health_checker = SystemHealthChecker()


async def get_health_status(detailed: bool = False) -> Dict[str, Any]:
    """获取健康状态的便捷函数"""
    return await health_checker.check_health(detailed=detailed)


async def check_readiness() -> bool:
    """检查系统是否准备就绪"""
    health = await health_checker.check_health()
    return health["status"] != HealthStatus.UNHEALTHY


async def check_liveness() -> bool:
    """检查系统是否存活"""
    # 简单的存活检查
    try:
        # 可以执行一些基本检查
        return True
    except Exception:
        return False