"""
API中间件
实现性能监控和频率限制
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from datetime import timedelta
from threading import Lock
from typing import Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

class PerformanceMetrics:
    """性能指标存储"""

    def __init__(self):
        self._lock = Lock()
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "max_response_time": 0.0,
            "min_response_time": float("inf"),
            "error_count": 0,
            "endpoint_metrics": defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0}),
        }

    def update(self, endpoint_key: str, response_time: float, is_error: bool) -> None:
        with self._lock:
            self.metrics["request_count"] += 1
            self.metrics["total_response_time"] += response_time
            self.metrics["max_response_time"] = max(self.metrics["max_response_time"], response_time)
            self.metrics["min_response_time"] = min(self.metrics["min_response_time"], response_time)
            if is_error:
                self.metrics["error_count"] += 1

            endpoint_metrics = self.metrics["endpoint_metrics"][endpoint_key]
            endpoint_metrics["count"] += 1
            endpoint_metrics["total_time"] += response_time
            if is_error:
                endpoint_metrics["errors"] += 1

    def snapshot(self) -> Dict:
        with self._lock:
            total_requests = self.metrics["request_count"]
            if total_requests == 0:
                return {
                    "total_requests": 0,
                    "average_response_time": 0.0,
                    "error_rate": 0.0,
                }
            return {
                "total_requests": total_requests,
                "average_response_time": self.metrics["total_response_time"] / total_requests,
                "max_response_time": self.metrics["max_response_time"],
                "min_response_time": self.metrics["min_response_time"]
                if self.metrics["min_response_time"] != float("inf")
                else 0.0,
                "error_count": self.metrics["error_count"],
                "error_rate": (self.metrics["error_count"] / total_requests) * 100,
                "endpoint_metrics": dict(self.metrics["endpoint_metrics"]),
            }

class PerformanceMonitoringMiddleware:
    """性能监控中间件"""

    def __init__(self, app: ASGIApp, metrics: PerformanceMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        start_time = time.perf_counter()
        path = request.url.path
        method = request.method
        endpoint_key = f"{method} {path}"

        logger.info(
            "API请求开始",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=request.client.host if request.client else "unknown",
        )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_time = time.perf_counter() - start_time
                self.metrics.update(endpoint_key, response_time, status_code >= 500)

                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Response-Time"] = f"{response_time:.3f}s"

                logger.info(
                    "API请求完成",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    response_time=response_time,
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            response_time = time.perf_counter() - start_time
            self.metrics.update(endpoint_key, response_time, True)
            logger.error(
                "API请求异常",
                request_id=request_id,
                method=method,
                path=path,
                error=str(e),
                response_time=response_time,
            )
            raise

class RateLimitingMiddleware:
    """API频率限制中间件"""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.client_hourly: Dict[str, deque] = defaultdict(deque)
        self._cleanup_task_started = False
        self._lock = Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self._cleanup_task_started:
            asyncio.create_task(self._cleanup_task())
            self._cleanup_task_started = True

        request = Request(scope, receive=receive)
        client_ip = self._get_client_ip(request)
        current_time = utc_now()

        if not self._check_rate_limit(client_ip, current_time):
            logger.warning(
                "API频率限制触发",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method,
            )
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "success": False,
                    "error": "请求频率过高，请稍后再试",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": 60,
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        self._record_request(client_ip, current_time)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                remaining = self._get_remaining_requests(client_ip, current_time)
                headers = MutableHeaders(scope=message)
                headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
                headers["X-RateLimit-Remaining"] = str(remaining)
            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str, current_time) -> bool:
        with self._lock:
            minute_ago = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
            hour_ago = current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

            client_minute_requests = self.client_requests[client_ip]
            while client_minute_requests and client_minute_requests[0] < minute_ago:
                client_minute_requests.popleft()

            client_hour_requests = self.client_hourly[client_ip]
            while client_hour_requests and client_hour_requests[0] < hour_ago:
                client_hour_requests.popleft()

            minute_limit = self.requests_per_minute + self.burst_size
            if len(client_minute_requests) >= minute_limit:
                return False

            if len(client_hour_requests) >= self.requests_per_hour:
                return False

            return True

    def _record_request(self, client_ip: str, current_time) -> None:
        with self._lock:
            self.client_requests[client_ip].append(current_time)
            self.client_hourly[client_ip].append(current_time)

    def _get_remaining_requests(self, client_ip: str, current_time) -> int:
        with self._lock:
            minute_ago = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
            client_requests = self.client_requests[client_ip]
            recent_requests = sum(1 for req_time in client_requests if req_time > minute_ago)
            return max(0, self.requests_per_minute - recent_requests)

    async def _cleanup_task(self):
        while True:
            await asyncio.sleep(60)
            self._cleanup_old_entries()

    def _cleanup_old_entries(self):
        current_time = utc_now()
        minute_ago = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
        hour_ago = current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

        expired_clients = []
        with self._lock:
            for client_ip in list(self.client_requests.keys()):
                client_requests = self.client_requests[client_ip]
                while client_requests and client_requests[0] < minute_ago:
                    client_requests.popleft()

                client_hourly = self.client_hourly[client_ip]
                while client_hourly and client_hourly[0] < hour_ago:
                    client_hourly.popleft()

                if not client_requests and not client_hourly:
                    expired_clients.append(client_ip)

            for client_ip in expired_clients:
                del self.client_requests[client_ip]
                del self.client_hourly[client_ip]

class MiddlewareManager:
    """中间件管理器"""

    def __init__(self):
        self.performance_metrics = PerformanceMetrics()

    def setup_middlewares(self, app):
        app.state.performance_metrics = self.performance_metrics
        app.add_middleware(PerformanceMonitoringMiddleware, metrics=self.performance_metrics)
        app.add_middleware(
            RateLimitingMiddleware,
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=10,
        )
        logger.info("所有API中间件已配置完成")

    def get_performance_metrics(self) -> Dict:
        return self.performance_metrics.snapshot()

middleware_manager = MiddlewareManager()
