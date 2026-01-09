"""
监控中间件
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send
from src.core.config import get_settings
from src.core.redis import get_redis
from .logger import (
    app_logger,
    request_logger,
    set_request_context,
    clear_request_context
)
from .metrics_collector import request_metrics
from .service import monitoring_service

class MonitoringMiddleware:
    """监控中间件"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        scope.setdefault("state", {})["request_id"] = request_id
        
        # 设置日志上下文
        set_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )
        
        # 记录请求开始
        start_time = time.perf_counter()
        await monitoring_service.performance_monitor.increment_active_requests()
        
        # 获取实验ID（如果有）
        experiment_id = None
        if "experiment_id" in request.path_params:
            experiment_id = request.path_params["experiment_id"]
        elif request.method == "GET":
            experiment_id = request.query_params.get("experiment_id")
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration = time.perf_counter() - start_time
                status_code = message["status"]

                request_logger.log_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration=duration,
                    request_id=request_id,
                    experiment_id=experiment_id,
                )

                await request_metrics.record_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration=duration,
                    experiment_id=experiment_id,
                )
                await monitoring_service.performance_monitor.record_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    duration=duration,
                )

                request_logger.log_slow_request(
                    method=request.method,
                    path=request.url.path,
                    duration=duration,
                    threshold=1.0,
                )

                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Response-Time"] = f"{duration:.3f}s"
            await send(message)

        try:
            settings = get_settings()
            rate_limit_paths = {
                "/api/v1/health": settings.HEALTH_RATE_LIMIT_PER_MINUTE,
                "/api/v1/mcp/health": settings.MCP_HEALTH_RATE_LIMIT_PER_MINUTE,
            }
            limit = rate_limit_paths.get(request.url.path)
            if limit:
                redis_client = get_redis()
                if redis_client:
                    try:
                        bucket = int(time.time() // 60)
                        key = f"rate:{request.client.host if request.client else 'unknown'}:{request.url.path}:{bucket}"
                        count = await redis_client.incr(key)
                        if count == 1:
                            await redis_client.expire(key, 65)
                        if count > limit:
                            duration = time.perf_counter() - start_time
                            await request_metrics.record_request(
                                method=request.method,
                                path=request.url.path,
                                status_code=429,
                                duration=duration,
                                experiment_id=experiment_id,
                            )
                            await monitoring_service.performance_monitor.record_request(
                                endpoint=request.url.path,
                                method=request.method,
                                status_code=429,
                                duration=duration,
                            )
                            response = JSONResponse(
                                status_code=429,
                                content={"detail": "Too Many Requests"},
                                headers={"X-Request-ID": request_id},
                            )
                            await response(scope, receive, send)
                            return
                    except Exception as exc:
                        app_logger.warning(
                            "健康检查限流失败",
                            request_id=request_id,
                            error=str(exc),
                        )

            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration = time.perf_counter() - start_time
            app_logger.error(
                f"Request failed: {request.method} {request.url.path}",
                exception=e,
                request_id=request_id,
                duration_ms=duration * 1000,
            )
            await request_metrics.record_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration=duration,
                experiment_id=experiment_id,
            )
            await monitoring_service.performance_monitor.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                duration=duration,
            )
            raise
        finally:
            await monitoring_service.performance_monitor.decrement_active_requests()
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
