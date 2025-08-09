"""
API中间件
实现性能监控、频率限制和错误处理
"""

import time
import uuid
import structlog
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import asyncio

logger = structlog.get_logger(__name__)

# ===== 性能监控中间件 =====

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    实现AC6: 接口响应时间监控和性能日志记录
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "max_response_time": 0.0,
            "min_response_time": float('inf'),
            "error_count": 0,
            "endpoint_metrics": defaultdict(lambda: {
                "count": 0,
                "total_time": 0.0,
                "errors": 0
            })
        }
    
    async def dispatch(self, request: Request, call_next):
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录开始时间
        start_time = time.time()
        
        # 提取路径和方法
        path = request.url.path
        method = request.method
        endpoint_key = f"{method} {path}"
        
        logger.info(
            "API请求开始",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        try:
            # 执行请求
            response = await call_next(request)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 更新指标
            self._update_metrics(endpoint_key, response_time, False)
            
            # 添加性能头部
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            
            logger.info(
                "API请求完成",
                request_id=request_id,
                method=method,
                path=path,
                status_code=response.status_code,
                response_time=response_time
            )
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # 更新错误指标
            self._update_metrics(endpoint_key, response_time, True)
            
            logger.error(
                "API请求异常",
                request_id=request_id,
                method=method,
                path=path,
                error=str(e),
                response_time=response_time
            )
            
            # 返回标准错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "内部服务器错误",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"X-Request-ID": request_id}
            )
    
    def _update_metrics(self, endpoint_key: str, response_time: float, is_error: bool):
        """更新性能指标"""
        self.metrics["request_count"] += 1
        self.metrics["total_response_time"] += response_time
        
        if response_time > self.metrics["max_response_time"]:
            self.metrics["max_response_time"] = response_time
        
        if response_time < self.metrics["min_response_time"]:
            self.metrics["min_response_time"] = response_time
        
        if is_error:
            self.metrics["error_count"] += 1
        
        # 更新端点指标
        endpoint_metrics = self.metrics["endpoint_metrics"][endpoint_key]
        endpoint_metrics["count"] += 1
        endpoint_metrics["total_time"] += response_time
        if is_error:
            endpoint_metrics["errors"] += 1
    
    def get_metrics(self) -> Dict:
        """获取性能指标"""
        total_requests = self.metrics["request_count"]
        if total_requests == 0:
            return {
                "total_requests": 0,
                "average_response_time": 0.0,
                "error_rate": 0.0
            }
        
        return {
            "total_requests": total_requests,
            "average_response_time": self.metrics["total_response_time"] / total_requests,
            "max_response_time": self.metrics["max_response_time"],
            "min_response_time": self.metrics["min_response_time"] if self.metrics["min_response_time"] != float('inf') else 0.0,
            "error_count": self.metrics["error_count"],
            "error_rate": (self.metrics["error_count"] / total_requests) * 100,
            "endpoint_metrics": dict(self.metrics["endpoint_metrics"])
        }

# ===== 频率限制中间件 =====

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    API频率限制中间件
    实现AC6: 配置API访问频率限制
    """
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        
        # 存储客户端请求记录
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.client_hourly: Dict[str, deque] = defaultdict(deque)
        
        # 清理任务将在需要时启动
        self._cleanup_task_started = False
    
    async def dispatch(self, request: Request, call_next):
        # 启动清理任务（仅一次）
        if not self._cleanup_task_started:
            asyncio.create_task(self._cleanup_task())
            self._cleanup_task_started = True
            
        client_ip = self._get_client_ip(request)
        current_time = datetime.now()
        
        # 检查频率限制
        if not self._check_rate_limit(client_ip, current_time):
            logger.warning(
                "API频率限制触发",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "success": False,
                    "error": "请求频率过高，请稍后再试",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # 记录请求
        self._record_request(client_ip, current_time)
        
        # 继续处理请求
        response = await call_next(request)
        
        # 添加频率限制头部
        remaining = self._get_remaining_requests(client_ip, current_time)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 检查X-Forwarded-For头部（代理情况）
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # 检查X-Real-IP头部
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # 使用直接连接IP
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str, current_time: datetime) -> bool:
        """检查是否超过频率限制"""
        minute_ago = current_time - timedelta(minutes=1)
        hour_ago = current_time - timedelta(hours=1)
        
        # 清理过期记录
        client_minute_requests = self.client_requests[client_ip]
        while client_minute_requests and client_minute_requests[0] < minute_ago:
            client_minute_requests.popleft()
        
        client_hour_requests = self.client_hourly[client_ip]
        while client_hour_requests and client_hour_requests[0] < hour_ago:
            client_hour_requests.popleft()
        
        # 检查限制
        if len(client_minute_requests) >= self.requests_per_minute:
            return False
        
        if len(client_hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    def _record_request(self, client_ip: str, current_time: datetime):
        """记录请求"""
        self.client_requests[client_ip].append(current_time)
        self.client_hourly[client_ip].append(current_time)
    
    def _get_remaining_requests(self, client_ip: str, current_time: datetime) -> int:
        """获取剩余请求数"""
        minute_ago = current_time - timedelta(minutes=1)
        client_requests = self.client_requests[client_ip]
        
        # 计算最近一分钟的请求数
        recent_requests = sum(1 for req_time in client_requests if req_time > minute_ago)
        
        return max(0, self.requests_per_minute - recent_requests)
    
    async def _cleanup_task(self):
        """定期清理过期数据"""
        while True:
            await asyncio.sleep(300)  # 每5分钟清理一次
            
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)
            
            # 清理过期的客户端记录
            expired_clients = []
            for client_ip in list(self.client_requests.keys()):
                # 清理分钟级记录
                client_requests = self.client_requests[client_ip]
                while client_requests and client_requests[0] < hour_ago:
                    client_requests.popleft()
                
                # 清理小时级记录
                client_hourly = self.client_hourly[client_ip]
                while client_hourly and client_hourly[0] < hour_ago:
                    client_hourly.popleft()
                
                # 如果没有记录了，标记为过期
                if not client_requests and not client_hourly:
                    expired_clients.append(client_ip)
            
            # 删除过期客户端
            for client_ip in expired_clients:
                del self.client_requests[client_ip]
                del self.client_hourly[client_ip]

# ===== 错误处理中间件 =====

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    统一错误处理中间件
    实现AC5: API接口的输入验证和错误响应标准化
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # FastAPI HTTPException已经是标准格式
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error": e.detail,
                    "error_code": "HTTP_ERROR",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except ValueError as e:
            # 验证错误
            logger.warning(
                "输入验证错误",
                error=str(e),
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": f"输入验证失败: {str(e)}",
                    "error_code": "VALIDATION_ERROR",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except Exception as e:
            # 未处理的异常
            logger.error(
                "未处理的API异常",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "内部服务器错误",
                    "error_code": "INTERNAL_ERROR",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )

# ===== 中间件管理器 =====

class MiddlewareManager:
    """中间件管理器"""
    
    def __init__(self):
        self.performance_middleware: Optional[PerformanceMonitoringMiddleware] = None
        self.rate_limit_middleware: Optional[RateLimitingMiddleware] = None
    
    def setup_middlewares(self, app):
        """设置所有中间件"""
        # 错误处理中间件（最外层）
        app.add_middleware(ErrorHandlingMiddleware)
        
        # 性能监控中间件
        self.performance_middleware = PerformanceMonitoringMiddleware(app)
        app.add_middleware(PerformanceMonitoringMiddleware)
        
        # 频率限制中间件
        self.rate_limit_middleware = RateLimitingMiddleware(
            app,
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=10
        )
        app.add_middleware(RateLimitingMiddleware,
                          requests_per_minute=60,
                          requests_per_hour=1000,
                          burst_size=10)
        
        logger.info("所有API中间件已配置完成")
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if self.performance_middleware:
            return self.performance_middleware.get_metrics()
        return {}

# 全局中间件管理器实例
middleware_manager = MiddlewareManager()