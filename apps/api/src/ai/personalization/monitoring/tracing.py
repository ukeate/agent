"""分布式追踪模块"""

import time
import json
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from opentelemetry import trace, metrics, baggage
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import asyncio

@dataclass
class SpanInfo:
    """Span信息"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    status: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]

class DistributedTracer:
    """分布式追踪器"""
    
    def __init__(self, 
                 service_name: str = "personalization-engine",
                 jaeger_host: str = "localhost",
                 jaeger_port: int = 6831):
        
        self.service_name = service_name
        
        # 创建资源
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # 初始化TracerProvider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer_provider = trace.get_tracer_provider()
        
        # 配置Jaeger导出器
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
            collector_endpoint=None,
            insecure=True
        )
        
        # 添加批处理器
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # 获取tracer
        self.tracer = trace.get_tracer(__name__)
        
        # 初始化传播器
        self.propagator = TraceContextTextMapPropagator()
        
        # 存储活跃的spans
        self.active_spans: Dict[str, SpanInfo] = {}
        
    def instrument_fastapi(self, app):
        """自动化FastAPI追踪"""
        FastAPIInstrumentor.instrument_app(app)
        
    def instrument_redis(self):
        """自动化Redis追踪"""
        RedisInstrumentor().instrument()
        
    def instrument_requests(self):
        """自动化HTTP请求追踪"""
        RequestsInstrumentor().instrument()
        
    @asynccontextmanager
    async def trace_async(self, 
                         operation: str,
                         attributes: Dict[str, Any] = None,
                         kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """异步追踪上下文管理器"""
        parent = trace.get_current_span()
        parent_ctx = parent.get_span_context() if parent else None
        parent_span_id = format(parent_ctx.span_id, '016x') if parent_ctx and parent_ctx.is_valid else None

        with self.tracer.start_as_current_span(
            operation,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            
            # 记录span信息
            span_context = span.get_span_context()
            span_info = SpanInfo(
                trace_id=format(span_context.trace_id, '032x'),
                span_id=format(span_context.span_id, '016x'),
                parent_span_id=parent_span_id,
                operation=operation,
                start_time=time.time(),
                end_time=None,
                duration=None,
                status="in_progress",
                attributes=attributes or {},
                events=[]
            )
            
            self.active_spans[span_info.span_id] = span_info
            
            try:
                yield span
                
                # 标记成功
                span.set_status(Status(StatusCode.OK))
                span_info.status = "success"
                
            except Exception as e:
                # 记录错误
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                span_info.status = "error"
                span_info.events.append({
                    "name": "exception",
                    "timestamp": time.time(),
                    "attributes": {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e)
                    }
                })
                raise
                
            finally:
                # 更新结束时间和持续时间
                span_info.end_time = time.time()
                span_info.duration = span_info.end_time - span_info.start_time
                
    @contextmanager
    def trace_sync(self,
                  operation: str,
                  attributes: Dict[str, Any] = None,
                  kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """同步追踪上下文管理器"""
        
        with self.tracer.start_as_current_span(
            operation,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                raise
                
    def trace_function(self, 
                      operation: Optional[str] = None,
                      attributes: Optional[Dict[str, Any]] = None):
        """函数追踪装饰器"""
        
        def decorator(func: Callable) -> Callable:
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.trace_async(op_name, attributes):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.trace_sync(op_name, attributes):
                        return func(*args, **kwargs)
                return sync_wrapper
                
        return decorator
        
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """添加事件到当前span"""
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes or {})
            
    def set_attribute(self, key: str, value: Any):
        """设置当前span的属性"""
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)
            
    def set_baggage(self, key: str, value: str):
        """设置baggage（跨服务传播的数据）"""
        baggage.set_baggage(key, value)
        
    def get_baggage(self, key: str) -> Optional[str]:
        """获取baggage"""
        return baggage.get_baggage(key)
        
    def inject_context(self, carrier: Dict[str, str]):
        """注入追踪上下文到载体（用于跨服务传播）"""
        inject(carrier)
        
    def extract_context(self, carrier: Dict[str, str]):
        """从载体提取追踪上下文"""
        return extract(carrier)
        
    def get_current_trace_id(self) -> Optional[str]:
        """获取当前trace ID"""
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            return format(span_context.trace_id, '032x')
        return None
        
    def get_current_span_id(self) -> Optional[str]:
        """获取当前span ID"""
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            return format(span_context.span_id, '016x')
        return None

class TracingMiddleware:
    """追踪中间件"""
    
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
        
    async def __call__(self, request, call_next):
        """处理请求追踪"""
        
        # 提取上游追踪上下文
        headers = dict(request.headers)
        context = self.tracer.extract_context(headers)
        
        # 创建span
        operation = f"{request.method} {request.url.path}"
        
        async with self.tracer.trace_async(
            operation,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": request.url.path,
                "http.user_agent": request.headers.get("user-agent", ""),
                "peer.ip": request.client.host if request.client else None
            },
            kind=trace.SpanKind.SERVER
        ) as span:
            
            # 记录请求开始时间
            start_time = time.time()
            
            try:
                # 处理请求
                response = await call_next(request)
                
                # 记录响应状态
                span.set_attribute("http.status_code", response.status_code)
                
                if response.status_code >= 400:
                    span.set_status(
                        Status(StatusCode.ERROR, f"HTTP {response.status_code}")
                    )
                    
                return response
                
            except Exception as e:
                # 记录异常
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                raise
                
            finally:
                # 记录请求持续时间
                duration = time.time() - start_time
                span.set_attribute("http.request.duration", duration)

class PerformanceTracer:
    """性能追踪器 - 专门用于追踪性能指标"""
    
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
        self.metrics: Dict[str, List[float]] = {}
        
    @asynccontextmanager
    async def trace_operation(self, 
                             operation: str,
                             component: str = "unknown"):
        """追踪操作性能"""
        
        start_time = time.perf_counter()
        
        async with self.tracer.trace_async(
            f"perf.{component}.{operation}",
            attributes={
                "component": component,
                "operation": operation
            }
        ) as span:
            
            try:
                yield span
                
            finally:
                # 计算延迟
                latency = (time.perf_counter() - start_time) * 1000  # ms
                
                # 记录延迟
                span.set_attribute("latency_ms", latency)
                
                # 添加到指标集合
                metric_key = f"{component}.{operation}"
                if metric_key not in self.metrics:
                    self.metrics[metric_key] = []
                self.metrics[metric_key].append(latency)
                
                # 记录延迟分级
                if latency < 10:
                    span.set_attribute("latency_level", "fast")
                elif latency < 100:
                    span.set_attribute("latency_level", "normal")
                elif latency < 500:
                    span.set_attribute("latency_level", "slow")
                else:
                    span.set_attribute("latency_level", "very_slow")
                    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """获取操作统计信息"""
        if operation not in self.metrics:
            return {}
            
        latencies = self.metrics[operation]
        
        import numpy as np
        
        return {
            "count": len(latencies),
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

class TraceAnalyzer:
    """追踪分析器"""
    
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
        self.traces: List[Dict[str, Any]] = []
        
    def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """分析单个trace"""
        
        # 获取trace中的所有spans
        spans = [
            span for span in self.tracer.active_spans.values()
            if span.trace_id == trace_id
        ]
        
        if not spans:
            return {}
            
        # 计算trace统计信息
        total_duration = max(s.duration for s in spans if s.duration)
        critical_path = self._find_critical_path(spans)
        
        # 分析性能瓶颈
        bottlenecks = self._identify_bottlenecks(spans)
        
        # 分析错误
        errors = [s for s in spans if s.status == "error"]
        
        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration": total_duration,
            "critical_path": critical_path,
            "critical_path_duration": sum(s.duration for s in critical_path if s.duration),
            "bottlenecks": bottlenecks,
            "error_count": len(errors),
            "errors": [
                {
                    "operation": e.operation,
                    "duration": e.duration,
                    "events": e.events
                }
                for e in errors
            ]
        }
        
    def _find_critical_path(self, spans: List[SpanInfo]) -> List[SpanInfo]:
        """找出关键路径"""
        
        # 简化版：找出最长的span序列
        # 实际应该构建依赖图并找出最长路径
        
        spans_sorted = sorted(spans, key=lambda s: s.duration or 0, reverse=True)
        return spans_sorted[:5]  # 返回前5个最耗时的spans
        
    def _identify_bottlenecks(self, spans: List[SpanInfo]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        
        bottlenecks = []
        
        for span in spans:
            if not span.duration:
                continue
                
            # 超过100ms的操作视为潜在瓶颈
            if span.duration > 0.1:
                bottlenecks.append({
                    "operation": span.operation,
                    "duration": span.duration * 1000,  # 转换为ms
                    "percentage": (span.duration / sum(s.duration or 0 for s in spans)) * 100
                })
                
        return sorted(bottlenecks, key=lambda b: b["duration"], reverse=True)
        
    def generate_trace_report(self, trace_id: str) -> str:
        """生成追踪报告"""
        
        analysis = self.analyze_trace(trace_id)
        
        if not analysis:
            return f"No trace found for ID: {trace_id}"
            
        report = [
            f"=== Trace Analysis Report ===",
            f"Trace ID: {trace_id}",
            f"Total Spans: {analysis['span_count']}",
            f"Total Duration: {analysis['total_duration']:.2f}s",
            f"Critical Path Duration: {analysis['critical_path_duration']:.2f}s",
            f"Error Count: {analysis['error_count']}",
            "",
            "=== Performance Bottlenecks ===",
        ]
        
        for bottleneck in analysis['bottlenecks'][:5]:
            report.append(
                f"- {bottleneck['operation']}: {bottleneck['duration']:.2f}ms ({bottleneck['percentage']:.1f}%)"
            )
            
        if analysis['errors']:
            report.append("")
            report.append("=== Errors ===")
            for error in analysis['errors']:
                report.append(f"- {error['operation']}: {error.get('duration', 0):.2f}s")
                
        return "\n".join(report)

# 全局追踪器实例
global_tracer = DistributedTracer()

# 装饰器便捷函数
def trace_async(operation: str = None, **kwargs):
    """异步函数追踪装饰器"""
    return global_tracer.trace_function(operation, kwargs)

def trace_sync(operation: str = None, **kwargs):
    """同步函数追踪装饰器"""
    return global_tracer.trace_function(operation, kwargs)

# 使用示例

@trace_async("feature_extraction")
async def extract_features(user_id: str) -> Dict[str, Any]:
    """提取用户特征示例"""
    
    # 添加属性
    global_tracer.set_attribute("user_id", user_id)
    
    # 模拟特征提取
    await asyncio.sleep(0.01)
    
    # 添加事件
    global_tracer.add_event("features_extracted", {
        "feature_count": 100
    })
    
    return {"features": [1, 2, 3]}

@trace_async("model_inference")
async def run_inference(features: Dict[str, Any]) -> List[float]:
    """运行模型推理示例"""
    
    global_tracer.set_attribute("feature_dim", len(features.get("features", [])))
    
    # 模拟推理
    await asyncio.sleep(0.02)
    
    return [0.9, 0.8, 0.7]

async def personalization_pipeline(user_id: str):
    """个性化流水线示例"""
    
    async with global_tracer.trace_async(
        "personalization_pipeline",
        {"user_id": user_id},
        kind=trace.SpanKind.SERVER
    ):
        
        # 提取特征
        features = await extract_features(user_id)
        
        # 运行推理
        scores = await run_inference(features)
        
        # 记录结果
        global_tracer.set_attribute("recommendation_count", len(scores))
        
        return scores
