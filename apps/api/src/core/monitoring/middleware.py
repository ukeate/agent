"""
监控中间件
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logger import (
    app_logger,
    request_logger,
    set_request_context,
    clear_request_context
)
from .metrics_collector import request_metrics


class MonitoringMiddleware(BaseHTTPMiddleware):
    """监控中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 设置日志上下文
        set_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )
        
        # 记录请求开始
        start_time = time.time()
        
        # 获取实验ID（如果有）
        experiment_id = None
        if "experiment_id" in request.path_params:
            experiment_id = request.path_params["experiment_id"]
        elif request.method == "GET":
            experiment_id = request.query_params.get("experiment_id")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算响应时间
            duration = time.time() - start_time
            
            # 记录请求日志
            request_logger.log_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                request_id=request_id,
                experiment_id=experiment_id
            )
            
            # 记录请求指标
            await request_metrics.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                experiment_id=experiment_id
            )
            
            # 记录慢请求
            request_logger.log_slow_request(
                method=request.method,
                path=request.url.path,
                duration=duration,
                threshold=1.0  # 1秒阈值
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # 记录错误
            app_logger.error(
                f"Request failed: {request.method} {request.url.path}",
                exception=e,
                request_id=request_id,
                duration_ms=duration * 1000
            )
            
            # 记录错误指标
            await request_metrics.record_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration=duration,
                experiment_id=experiment_id
            )
            
            raise
        finally:
            # 清理日志上下文
            clear_request_context()


class LoggingRoute(APIRoute):
    """带日志的路由"""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def logging_route_handler(request: Request) -> Response:
            # 获取请求ID
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            # 设置日志上下文
            set_request_context(
                request_id=request_id,
                endpoint=self.endpoint.__name__ if self.endpoint else "unknown"
            )
            
            # 记录API调用
            app_logger.debug(
                f"API call: {self.endpoint.__name__ if self.endpoint else 'unknown'}",
                endpoint=self.endpoint.__name__ if self.endpoint else "unknown",
                path_params=request.path_params,
                query_params=dict(request.query_params)
            )
            
            try:
                response = await original_route_handler(request)
                return response
            finally:
                clear_request_context()
        
        return logging_route_handler


class HealthCheckMiddleware:
    """健康检查中间件"""
    
    def __init__(self, app: ASGIApp, health_check_path: str = "/health"):
        self.app = app
        self.health_check_path = health_check_path
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == self.health_check_path:
            # 返回健康状态
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"status": "healthy"}',
            })
        else:
            await self.app(scope, receive, send)


class MetricsMiddleware:
    """指标暴露中间件"""
    
    def __init__(self, app: ASGIApp, metrics_path: str = "/metrics"):
        self.app = app
        self.metrics_path = metrics_path
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == self.metrics_path:
            # 导入指标收集器
            from .metrics_collector import metrics_collector
            
            # 获取所有指标
            metrics = await metrics_collector.get_all_metrics()
            
            # 格式化为Prometheus格式
            prometheus_metrics = self._format_prometheus_metrics(metrics)
            
            # 返回指标
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": prometheus_metrics.encode(),
            })
        else:
            await self.app(scope, receive, send)
    
    def _format_prometheus_metrics(self, metrics: dict) -> str:
        """格式化为Prometheus格式"""
        lines = []
        
        for metric_name, metric_data in metrics.items():
            # 添加帮助信息
            lines.append(f"# HELP {metric_name} {metric_name}")
            lines.append(f"# TYPE {metric_name} gauge")
            
            for labels, values in metric_data.items():
                if labels:
                    label_str = "{" + labels + "}"
                else:
                    label_str = ""
                
                # 添加指标值
                if isinstance(values, dict):
                    for stat_name, stat_value in values.items():
                        lines.append(f"{metric_name}_{stat_name}{label_str} {stat_value}")
                else:
                    lines.append(f"{metric_name}{label_str} {values}")
        
        return "\n".join(lines) + "\n"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 获取当前时间
        current_time = time.time()
        
        # 清理旧记录
        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                t for t in self.request_times[client_ip]
                if current_time - t < 60
            ]
        else:
            self.request_times[client_ip] = []
        
        # 检查速率限制
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            app_logger.warning(
                f"Rate limit exceeded for {client_ip}",
                client_ip=client_ip,
                requests_count=len(self.request_times[client_ip])
            )
            
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # 记录请求时间
        self.request_times[client_ip].append(current_time)
        
        # 继续处理请求
        response = await call_next(request)
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except ValueError as e:
            app_logger.warning(f"Validation error: {str(e)}", exception=e)
            return Response(
                content=f'{{"error": "Validation error: {str(e)}"}}',
                status_code=400,
                media_type="application/json"
            )
        except PermissionError as e:
            app_logger.warning(f"Permission denied: {str(e)}", exception=e)
            return Response(
                content=f'{{"error": "Permission denied: {str(e)}"}}',
                status_code=403,
                media_type="application/json"
            )
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            app_logger.error(
                f"Unhandled exception in request {request_id}",
                exception=e,
                request_id=request_id
            )
            return Response(
                content='{"error": "Internal server error"}',
                status_code=500,
                media_type="application/json"
            )