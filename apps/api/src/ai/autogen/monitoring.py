"""
企业级AutoGen监控和调试工具
实现智能体性能监控、对话追踪、调试功能和OpenTelemetry集成
"""
import asyncio
import json
import time
import traceback
import uuid
import hashlib
import platform
import os
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import structlog

# OpenTelemetry 导入
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import get_tracer, Status, StatusCode
    from opentelemetry.metrics import get_meter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None
    logger = structlog.get_logger(__name__)
    logger.warning("OpenTelemetry不可用，将使用内置监控功能")

from .events import Event, EventType, EventHandler, EventBus, EventPriority
from .async_manager import AsyncAgentManager, AgentTask, TaskStatus
from .event_processors import AsyncEventProcessingEngine, EventProcessor, EventContext, ProcessingResult
from .event_store import EventStore, EventReplayService
from .event_router import EventRouter, EventFilter
from .distributed_events import DistributedEventCoordinator

logger = structlog.get_logger(__name__)


class DebugLevel(str, Enum):
    """调试级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceType(str, Enum):
    """追踪类型"""
    CONVERSATION = "conversation"
    TASK_EXECUTION = "task_execution"
    AGENT_LIFECYCLE = "agent_lifecycle"
    EVENT_FLOW = "event_flow"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=lambda: utc_now())
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class TraceSpan:
    """追踪跨度"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    trace_id: str = ""
    operation_name: str = ""
    start_time: datetime = field(default_factory=lambda: utc_now())
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = "completed") -> None:
        """完成跨度"""
        self.end_time = utc_now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
    
    def add_log(self, level: str, message: str, **kwargs) -> None:
        """添加日志"""
        self.logs.append({
            "timestamp": utc_now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs
        }


@dataclass
class ConversationTrace:
    """对话追踪"""
    conversation_id: str
    session_id: str
    start_time: datetime = field(default_factory=lambda: utc_now())
    end_time: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    message_count: int = 0
    spans: List[TraceSpan] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: TraceSpan) -> None:
        """添加跨度"""
        span.trace_id = self.conversation_id
        self.spans.append(span)
    
    def add_event(self, event: Event) -> None:
        """添加事件"""
        self.events.append(event)
        if event.type == EventType.MESSAGE_SENT:
            self.message_count += 1
    
    def finish(self) -> None:
        """完成追踪"""
        self.end_time = utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000 if self.end_time else None,
            "participants": self.participants,
            "message_count": self.message_count,
            "span_count": len(self.spans),
            "event_count": len(self.events),
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics: Dict[str, List[PerformanceMetric]] = {}
        self.active_measurements: Dict[str, float] = {}
        self.performance_thresholds = {
            "response_time": 5000,  # 5秒
            "memory_usage": 512 * 1024 * 1024,  # 512MB
            "cpu_usage": 80.0,  # 80%
            "task_queue_size": 100
        }
        
        logger.info("性能监控器初始化完成")
    
    def start_measurement(self, measurement_id: str) -> None:
        """开始测量"""
        self.active_measurements[measurement_id] = time.time()
    
    def end_measurement(
        self,
        measurement_id: str,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """结束测量"""
        if measurement_id not in self.active_measurements:
            return None
        
        start_time = self.active_measurements.pop(measurement_id)
        duration = (time.time() - start_time) * 1000  # 转换为毫秒
        
        self.record_metric(
            name=metric_name,
            value=duration,
            unit="ms",
            tags=tags or {}
        )
        
        return duration
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录指标"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # 保持最近1000个数据点
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # 检查性能阈值
        self._check_performance_threshold(metric)
        
        logger.debug(f"记录性能指标: {name}={value}{unit}", tags=tags)
    
    def _check_performance_threshold(self, metric: PerformanceMetric) -> None:
        """检查性能阈值"""
        threshold = self.performance_thresholds.get(metric.name)
        if threshold and metric.value > threshold:
            asyncio.create_task(self._emit_performance_alert(metric, threshold))
    
    async def _emit_performance_alert(
        self,
        metric: PerformanceMetric,
        threshold: float
    ) -> None:
        """发出性能告警"""
        await self.event_bus.publish(Event(
            type=EventType.ERROR_OCCURRED,
            source="performance_monitor",
            data={
                "type": "performance_threshold_exceeded",
                "metric": metric.to_dict(),
                "threshold": threshold,
                "severity": "warning"
            }
        ))
    
    def get_metric_summary(
        self,
        name: str,
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """获取指标摘要"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = utc_now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics[name]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "unit": recent_metrics[-1].unit,
            "duration_minutes": duration_minutes,
            "threshold": self.performance_thresholds.get(name)
        }
    
    def get_all_metrics_summary(
        self,
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """获取所有指标摘要"""
        summaries = {}
        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name, duration_minutes)
            if summary:
                summaries[metric_name] = summary
        
        return {
            "summaries": summaries,
            "total_metrics": len(self.metrics),
            "active_measurements": len(self.active_measurements),
            "thresholds": self.performance_thresholds
        }


class ConversationTracker:
    """对话追踪器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.active_traces: Dict[str, ConversationTrace] = {}
        self.completed_traces: List[ConversationTrace] = []
        self.max_completed_traces = 1000
        
        logger.info("对话追踪器初始化完成")
    
    def start_conversation_trace(
        self,
        conversation_id: str,
        session_id: str,
        participants: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTrace:
        """开始对话追踪"""
        trace = ConversationTrace(
            conversation_id=conversation_id,
            session_id=session_id,
            participants=participants.copy(),
            metadata=metadata or {}
        )
        
        self.active_traces[conversation_id] = trace
        
        logger.info(
            "开始对话追踪",
            conversation_id=conversation_id,
            participants=participants
        )
        
        return trace
    
    def end_conversation_trace(self, conversation_id: str) -> Optional[ConversationTrace]:
        """结束对话追踪"""
        if conversation_id not in self.active_traces:
            return None
        
        trace = self.active_traces.pop(conversation_id)
        trace.finish()
        
        self.completed_traces.append(trace)
        
        # 保持已完成追踪数量在限制内
        if len(self.completed_traces) > self.max_completed_traces:
            self.completed_traces = self.completed_traces[-self.max_completed_traces:]
        
        logger.info(
            "结束对话追踪",
            conversation_id=conversation_id,
            duration_ms=trace.to_dict().get("duration_ms"),
            message_count=trace.message_count
        )
        
        return trace
    
    def add_span_to_conversation(
        self,
        conversation_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[TraceSpan]:
        """为对话添加跨度"""
        if conversation_id not in self.active_traces:
            return None
        
        span = TraceSpan(
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        self.active_traces[conversation_id].add_span(span)
        
        return span
    
    def finish_span(
        self,
        conversation_id: str,
        span_id: str,
        status: str = "completed"
    ) -> bool:
        """完成跨度"""
        if conversation_id not in self.active_traces:
            return False
        
        trace = self.active_traces[conversation_id]
        for span in trace.spans:
            if span.span_id == span_id:
                span.finish(status)
                return True
        
        return False
    
    def add_event_to_conversation(
        self,
        conversation_id: str,
        event: Event
    ) -> bool:
        """为对话添加事件"""
        if conversation_id not in self.active_traces:
            return False
        
        self.active_traces[conversation_id].add_event(event)
        return True
    
    def get_conversation_trace(
        self,
        conversation_id: str
    ) -> Optional[ConversationTrace]:
        """获取对话追踪"""
        # 先检查活跃追踪
        if conversation_id in self.active_traces:
            return self.active_traces[conversation_id]
        
        # 再检查已完成追踪
        for trace in self.completed_traces:
            if trace.conversation_id == conversation_id:
                return trace
        
        return None
    
    def search_traces(
        self,
        session_id: Optional[str] = None,
        participant: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ConversationTrace]:
        """搜索追踪"""
        results = []
        all_traces = list(self.active_traces.values()) + self.completed_traces
        
        for trace in all_traces:
            # 应用过滤条件
            if session_id and trace.session_id != session_id:
                continue
            if participant and participant not in trace.participants:
                continue
            if start_time and trace.start_time < start_time:
                continue
            if end_time and trace.start_time > end_time:
                continue
            
            results.append(trace)
            
            # 限制结果数量
            if len(results) >= limit:
                break
        
        return results
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """获取追踪统计"""
        active_count = len(self.active_traces)
        completed_count = len(self.completed_traces)
        
        # 计算平均对话时长
        avg_duration = 0
        if self.completed_traces:
            total_duration = sum(
                (trace.end_time - trace.start_time).total_seconds()
                for trace in self.completed_traces
                if trace.end_time
            )
            avg_duration = total_duration / len(self.completed_traces)
        
        # 计算平均消息数
        avg_messages = 0
        if self.completed_traces:
            total_messages = sum(trace.message_count for trace in self.completed_traces)
            avg_messages = total_messages / len(self.completed_traces)
        
        return {
            "active_traces": active_count,
            "completed_traces": completed_count,
            "total_traces": active_count + completed_count,
            "avg_duration_seconds": avg_duration,
            "avg_messages_per_conversation": avg_messages
        }


class DebugConsole:
    """调试控制台"""
    
    def __init__(
        self,
        event_bus: EventBus,
        agent_manager: AsyncAgentManager,
        performance_monitor: PerformanceMonitor,
        conversation_tracker: ConversationTracker
    ):
        self.event_bus = event_bus
        self.agent_manager = agent_manager
        self.performance_monitor = performance_monitor
        self.conversation_tracker = conversation_tracker
        
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.breakpoints: Dict[str, Callable] = {}
        self.debug_level = DebugLevel.INFO
        
        logger.info("调试控制台初始化完成")
    
    def start_debug_session(
        self,
        session_id: str,
        target_type: str,
        target_id: str,
        debug_level: DebugLevel = DebugLevel.DEBUG
    ) -> Dict[str, Any]:
        """开始调试会话"""
        session = {
            "session_id": session_id,
            "target_type": target_type,
            "target_id": target_id,
            "debug_level": debug_level,
            "start_time": utc_now(),
            "events": [],
            "snapshots": [],
            "active": True
        }
        
        self.debug_sessions[session_id] = session
        
        logger.info(
            "开始调试会话",
            session_id=session_id,
            target_type=target_type,
            target_id=target_id
        )
        
        return session
    
    def stop_debug_session(self, session_id: str) -> bool:
        """停止调试会话"""
        if session_id not in self.debug_sessions:
            return False
        
        session = self.debug_sessions[session_id]
        session["active"] = False
        session["end_time"] = utc_now()
        
        logger.info("停止调试会话", session_id=session_id)
        return True
    
    def add_breakpoint(
        self,
        breakpoint_id: str,
        condition: Callable[[Event], bool]
    ) -> None:
        """添加断点"""
        self.breakpoints[breakpoint_id] = condition
        logger.info("添加断点", breakpoint_id=breakpoint_id)
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """移除断点"""
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            logger.info("移除断点", breakpoint_id=breakpoint_id)
            return True
        return False
    
    async def capture_agent_snapshot(self, agent_id: str) -> Dict[str, Any]:
        """捕获智能体快照"""
        try:
            agent_info = await self.agent_manager.get_agent_info(agent_id)
            agent_tasks = await self.agent_manager.list_tasks(agent_id=agent_id)
            
            snapshot = {
                "timestamp": utc_now().isoformat(),
                "agent_id": agent_id,
                "agent_info": agent_info,
                "tasks": agent_tasks,
                "performance_metrics": self.performance_monitor.get_all_metrics_summary(5),
                "stack_trace": self._get_current_stack_trace()
            }
            
            return snapshot
            
        except Exception as e:
            logger.error("捕获智能体快照失败", agent_id=agent_id, error=str(e))
            return {"error": str(e)}
    
    def _get_current_stack_trace(self) -> List[str]:
        """获取当前堆栈跟踪"""
        return traceback.format_stack()
    
    async def execute_debug_command(
        self,
        session_id: str,
        command: str,
        args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行调试命令"""
        if session_id not in self.debug_sessions:
            return {"error": "调试会话不存在"}
        
        session = self.debug_sessions[session_id]
        if not session["active"]:
            return {"error": "调试会话已停止"}
        
        args = args or {}
        
        try:
            if command == "snapshot":
                target_id = args.get("target_id", session["target_id"])
                snapshot = await self.capture_agent_snapshot(target_id)
                session["snapshots"].append(snapshot)
                return {"result": snapshot}
            
            elif command == "metrics":
                metrics = self.performance_monitor.get_all_metrics_summary(
                    duration_minutes=args.get("duration", 60)
                )
                return {"result": metrics}
            
            elif command == "traces":
                traces = self.conversation_tracker.search_traces(
                    session_id=args.get("session_id"),
                    limit=args.get("limit", 10)
                )
                return {"result": [trace.to_dict() for trace in traces]}
            
            elif command == "agent_status":
                agents = await self.agent_manager.list_agents()
                return {"result": agents}
            
            elif command == "tasks":
                tasks = await self.agent_manager.list_tasks(
                    agent_id=args.get("agent_id"),
                    status=TaskStatus(args["status"]) if args.get("status") else None
                )
                return {"result": tasks}
            
            else:
                return {"error": f"未知调试命令: {command}"}
                
        except Exception as e:
            logger.error("执行调试命令失败", command=command, error=str(e))
            return {"error": str(e)}
    
    def get_debug_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取调试会话信息"""
        if session_id not in self.debug_sessions:
            return None
        
        session = self.debug_sessions[session_id].copy()
        session["event_count"] = len(session["events"])
        session["snapshot_count"] = len(session["snapshots"])
        
        return session
    
    def list_debug_sessions(self) -> List[Dict[str, Any]]:
        """列出所有调试会话"""
        sessions = []
        for session_id, session in self.debug_sessions.items():
            session_info = {
                "session_id": session_id,
                "target_type": session["target_type"],
                "target_id": session["target_id"],
                "active": session["active"],
                "start_time": session["start_time"].isoformat(),
                "event_count": len(session["events"])
            }
            sessions.append(session_info)
        
        return sessions


class AgentDashboard:
    """智能体仪表板"""
    
    def __init__(
        self,
        agent_manager: AsyncAgentManager,
        performance_monitor: PerformanceMonitor,
        conversation_tracker: ConversationTracker,
        event_bus: EventBus
    ):
        self.agent_manager = agent_manager
        self.performance_monitor = performance_monitor
        self.conversation_tracker = conversation_tracker
        self.event_bus = event_bus
        
        logger.info("智能体仪表板初始化完成")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        try:
            # 获取智能体状态
            agents = await self.agent_manager.list_agents()
            manager_stats = self.agent_manager.get_manager_stats()
            
            # 获取性能指标
            performance_data = self.performance_monitor.get_all_metrics_summary()
            
            # 获取对话统计
            conversation_stats = self.conversation_tracker.get_trace_statistics()
            
            # 获取事件总线状态
            event_stats = self.event_bus.get_stats()
            
            # 获取任务统计
            all_tasks = await self.agent_manager.list_tasks()
            task_stats = self._calculate_task_stats(all_tasks)
            
            return {
                "timestamp": utc_now().isoformat(),
                "agents": {
                    "list": agents,
                    "stats": manager_stats
                },
                "performance": performance_data,
                "conversations": conversation_stats,
                "events": event_stats,
                "tasks": task_stats,
                "system_health": await self._get_system_health()
            }
            
        except Exception as e:
            logger.error("获取仪表板数据失败", error=str(e))
            return {"error": str(e)}
    
    def _calculate_task_stats(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算任务统计"""
        total_tasks = len(tasks)
        status_counts = {}
        
        for task in tasks:
            status = task.get("status", "unknown")
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        # 计算成功率
        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0
        
        return {
            "total": total_tasks,
            "by_status": status_counts,
            "success_rate": success_rate
        }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 检查关键指标
            health_status = "healthy"
            issues = []
            
            # 检查错误率
            performance_data = self.performance_monitor.get_all_metrics_summary(60)
            error_metrics = performance_data.get("summaries", {}).get("error_rate")
            if error_metrics and error_metrics.get("avg", 0) > 0.1:
                health_status = "degraded"
                issues.append("错误率过高")
            
            # 检查响应时间
            response_time_metrics = performance_data.get("summaries", {}).get("response_time")
            if response_time_metrics and response_time_metrics.get("avg", 0) > 5000:
                health_status = "degraded"
                issues.append("响应时间过长")
            
            # 检查任务队列
            manager_stats = self.agent_manager.get_manager_stats()
            if manager_stats.get("tasks", {}).get("pending", 0) > 50:
                health_status = "degraded"
                issues.append("任务队列积压")
            
            return {
                "status": health_status,
                "issues": issues,
                "last_check": utc_now().isoformat()
            }
            
        except Exception as e:
            logger.error("系统健康检查失败", error=str(e))
            return {
                "status": "unknown",
                "issues": [f"健康检查失败: {str(e)}"],
                "last_check": utc_now().isoformat()
            }


class MonitoringEventHandler(EventHandler):
    """监控事件处理器"""
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        conversation_tracker: ConversationTracker
    ):
        self.performance_monitor = performance_monitor
        self.conversation_tracker = conversation_tracker
    
    @property
    def supported_events(self) -> List[EventType]:
        return list(EventType)
    
    async def handle(self, event: Event) -> None:
        """处理事件"""
        try:
            # 记录事件处理性能指标
            if event.type == EventType.TASK_COMPLETED:
                execution_time = event.data.get("execution_time", 0)
                self.performance_monitor.record_metric(
                    "task_execution_time",
                    execution_time * 1000,  # 转换为毫秒
                    "ms",
                    {"agent_id": event.source, "task_type": event.data.get("task_type")}
                )
            
            # 添加事件到对话追踪
            if event.conversation_id:
                self.conversation_tracker.add_event_to_conversation(
                    event.conversation_id, event
                )
            
        except Exception as e:
            logger.error("监控事件处理失败", event_type=event.type, error=str(e))


class EventProcessingMonitor:
    """事件处理监控器"""
    
    def __init__(
        self,
        processing_engine: Optional[AsyncEventProcessingEngine] = None,
        event_store: Optional[EventStore] = None,
        event_router: Optional[EventRouter] = None,
        distributed_coordinator: Optional[DistributedEventCoordinator] = None
    ):
        self.processing_engine = processing_engine
        self.event_store = event_store
        self.event_router = event_router
        self.distributed_coordinator = distributed_coordinator
        
        # 监控指标
        self.event_counts = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "retried_events": 0
        }
        
        # 延迟统计
        self.latency_stats = {
            "processing_latency": [],
            "routing_latency": [],
            "storage_latency": []
        }
        
        logger.info("事件处理监控器初始化完成")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """获取事件处理统计"""
        stats = {
            "event_counts": self.event_counts.copy(),
            "latency_stats": self._calculate_latency_stats()
        }
        
        # 从处理引擎获取统计
        if self.processing_engine:
            engine_stats = self.processing_engine.get_stats()
            stats["engine"] = engine_stats
        
        # 从事件存储获取统计
        if self.event_store:
            store_stats = self.event_store.get_stats()
            stats["storage"] = store_stats
        
        # 从路由器获取统计
        if self.event_router:
            router_stats = self.event_router.get_stats()
            stats["router"] = router_stats
        
        # 从分布式协调器获取统计
        if self.distributed_coordinator:
            cluster_status = await self.distributed_coordinator.get_cluster_status()
            stats["cluster"] = cluster_status
        
        return stats
    
    def _calculate_latency_stats(self) -> Dict[str, Any]:
        """计算延迟统计"""
        result = {}
        
        for latency_type, values in self.latency_stats.items():
            if values:
                # 保留最近1000个值
                recent_values = values[-1000:]
                result[latency_type] = {
                    "count": len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "avg": sum(recent_values) / len(recent_values),
                    "p50": self._percentile(recent_values, 50),
                    "p95": self._percentile(recent_values, 95),
                    "p99": self._percentile(recent_values, 99)
                }
            else:
                result[latency_type] = {
                    "count": 0,
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0
                }
        
        return result
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def record_event_processing(
        self,
        event_type: str,
        processing_time_ms: float,
        success: bool,
        retried: bool = False
    ) -> None:
        """记录事件处理"""
        self.event_counts["total_events"] += 1
        
        if success:
            self.event_counts["processed_events"] += 1
        else:
            self.event_counts["failed_events"] += 1
        
        if retried:
            self.event_counts["retried_events"] += 1
        
        # 记录处理延迟
        self.latency_stats["processing_latency"].append(processing_time_ms)
        
        # 保持列表大小
        if len(self.latency_stats["processing_latency"]) > 10000:
            self.latency_stats["processing_latency"] = self.latency_stats["processing_latency"][-10000:]
    
    def record_routing_latency(self, latency_ms: float) -> None:
        """记录路由延迟"""
        self.latency_stats["routing_latency"].append(latency_ms)
        
        if len(self.latency_stats["routing_latency"]) > 10000:
            self.latency_stats["routing_latency"] = self.latency_stats["routing_latency"][-10000:]
    
    def record_storage_latency(self, latency_ms: float) -> None:
        """记录存储延迟"""
        self.latency_stats["storage_latency"].append(latency_ms)
        
        if len(self.latency_stats["storage_latency"]) > 10000:
            self.latency_stats["storage_latency"] = self.latency_stats["storage_latency"][-10000:]
    
    async def get_event_flow_visualization(self) -> Dict[str, Any]:
        """获取事件流可视化数据"""
        visualization = {
            "timestamp": utc_now().isoformat(),
            "nodes": [],
            "edges": [],
            "metrics": {}
        }
        
        # 添加节点信息
        if self.distributed_coordinator:
            cluster_status = await self.distributed_coordinator.get_cluster_status()
            for node_id, node_info in cluster_status.get("nodes", {}).items():
                visualization["nodes"].append({
                    "id": node_id,
                    "type": "processing_node",
                    "status": node_info["status"],
                    "load": node_info["load"]
                })
        
        # 添加处理器信息
        if self.processing_engine:
            engine_stats = self.processing_engine.get_stats()
            for processor_name, processor_stats in engine_stats.get("processor_metrics", {}).items():
                visualization["nodes"].append({
                    "id": processor_name,
                    "type": "processor",
                    "processed": processor_stats.get("processed", 0),
                    "success_rate": processor_stats.get("success_rate", 0)
                })
        
        # 添加路由信息
        if self.event_router:
            router_stats = self.event_router.get_stats()
            for route_stat in router_stats.get("route_stats", []):
                visualization["edges"].append({
                    "from": "event_source",
                    "to": route_stat["name"],
                    "type": "route",
                    "enabled": route_stat["enabled"]
                })
        
        # 添加整体指标
        visualization["metrics"] = await self.get_processing_stats()
        
        return visualization
    
    async def analyze_event_patterns(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """分析事件模式"""
        analysis = {
            "time_window_minutes": time_window_minutes,
            "patterns": [],
            "anomalies": [],
            "recommendations": []
        }
        
        # 分析事件频率模式
        if self.processing_engine:
            engine_stats = self.processing_engine.get_stats()
            events_processed = engine_stats.get("events_processed", 0)
            events_failed = engine_stats.get("events_failed", 0)
            
            # 计算失败率
            failure_rate = events_failed / events_processed if events_processed > 0 else 0
            
            if failure_rate > 0.1:
                analysis["anomalies"].append({
                    "type": "high_failure_rate",
                    "value": failure_rate,
                    "threshold": 0.1,
                    "severity": "high"
                })
                analysis["recommendations"].append(
                    "高失败率检测到，建议检查事件处理器配置和错误日志"
                )
        
        # 分析延迟模式
        latency_stats = self._calculate_latency_stats()
        for latency_type, stats in latency_stats.items():
            if stats["p99"] > 1000:  # P99延迟超过1秒
                analysis["anomalies"].append({
                    "type": f"high_{latency_type}",
                    "p99": stats["p99"],
                    "threshold": 1000,
                    "severity": "medium"
                })
                analysis["recommendations"].append(
                    f"{latency_type}的P99延迟过高，建议优化处理逻辑或增加处理资源"
                )
        
        # 分析事件分布模式
        if self.event_router:
            router_stats = self.event_router.get_stats()
            total_routed = router_stats.get("events_routed", 0)
            default_used = router_stats.get("default_routes_used", 0)
            
            if total_routed > 0:
                default_ratio = default_used / total_routed
                if default_ratio > 0.5:
                    analysis["patterns"].append({
                        "type": "high_default_routing",
                        "ratio": default_ratio,
                        "description": "超过50%的事件使用默认路由"
                    })
                    analysis["recommendations"].append(
                        "大量事件使用默认路由，建议添加更多特定路由规则"
                    )
        
        return analysis


class EventDebugger:
    """事件调试器"""
    
    def __init__(
        self,
        event_bus: EventBus,
        processing_engine: Optional[AsyncEventProcessingEngine] = None,
        event_store: Optional[EventStore] = None
    ):
        self.event_bus = event_bus
        self.processing_engine = processing_engine
        self.event_store = event_store
        
        # 调试配置
        self.debug_enabled = False
        self.event_breakpoints = {}
        self.event_filters = []
        self.captured_events = []
        self.max_captured_events = 1000
        
        logger.info("事件调试器初始化完成")
    
    def enable_debug(self) -> None:
        """启用调试模式"""
        self.debug_enabled = True
        logger.info("事件调试模式已启用")
    
    def disable_debug(self) -> None:
        """禁用调试模式"""
        self.debug_enabled = False
        logger.info("事件调试模式已禁用")
    
    def add_event_breakpoint(
        self,
        breakpoint_id: str,
        event_type: EventType,
        condition: Optional[Callable[[Event], bool]] = None
    ) -> None:
        """添加事件断点"""
        self.event_breakpoints[breakpoint_id] = {
            "event_type": event_type,
            "condition": condition or (lambda e: True),
            "hit_count": 0
        }
        logger.info(f"添加事件断点", breakpoint_id=breakpoint_id, event_type=event_type)
    
    def remove_event_breakpoint(self, breakpoint_id: str) -> bool:
        """移除事件断点"""
        if breakpoint_id in self.event_breakpoints:
            del self.event_breakpoints[breakpoint_id]
            logger.info(f"移除事件断点", breakpoint_id=breakpoint_id)
            return True
        return False
    
    async def check_breakpoint(self, event: Event) -> bool:
        """检查断点"""
        if not self.debug_enabled:
            return False
        
        for breakpoint_id, breakpoint in self.event_breakpoints.items():
            if event.type == breakpoint["event_type"]:
                if breakpoint["condition"](event):
                    breakpoint["hit_count"] += 1
                    
                    # 捕获事件
                    await self.capture_event(event, breakpoint_id)
                    
                    logger.info(
                        "事件断点命中",
                        breakpoint_id=breakpoint_id,
                        event_type=event.type,
                        hit_count=breakpoint["hit_count"]
                    )
                    
                    return True
        
        return False
    
    async def capture_event(
        self,
        event: Event,
        breakpoint_id: Optional[str] = None
    ) -> None:
        """捕获事件"""
        capture_data = {
            "timestamp": utc_now().isoformat(),
            "event": event.to_dict() if hasattr(event, 'to_dict') else str(event),
            "breakpoint_id": breakpoint_id,
            "stack_trace": traceback.format_stack()
        }
        
        # 如果有处理引擎，获取当前状态
        if self.processing_engine:
            capture_data["engine_stats"] = self.processing_engine.get_stats()
        
        self.captured_events.append(capture_data)
        
        # 限制捕获数量
        if len(self.captured_events) > self.max_captured_events:
            self.captured_events = self.captured_events[-self.max_captured_events:]
    
    async def replay_event(
        self,
        event_id: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """重播事件"""
        try:
            # 从存储获取事件
            if self.event_store:
                event = await self.event_store.get_event(event_id)
                if not event:
                    return {"error": f"事件未找到: {event_id}"}
            else:
                return {"error": "事件存储未配置"}
            
            # 应用修改
            if modifications:
                for key, value in modifications.items():
                    if hasattr(event, key):
                        setattr(event, key, value)
            
            # 标记为重播事件
            if hasattr(event, 'data') and isinstance(event.data, dict):
                event.data["is_replay"] = True
                event.data["replay_time"] = utc_now().isoformat()
            
            # 重新发布事件
            success = await self.event_bus.publish(event)
            
            return {
                "success": success,
                "event_id": event_id,
                "modifications": modifications
            }
            
        except Exception as e:
            logger.error(f"事件重播失败", event_id=event_id, error=str(e))
            return {"error": str(e)}
    
    def get_captured_events(
        self,
        limit: int = 100,
        breakpoint_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取捕获的事件"""
        events = self.captured_events
        
        # 按断点过滤
        if breakpoint_id:
            events = [e for e in events if e.get("breakpoint_id") == breakpoint_id]
        
        # 限制返回数量
        return events[-limit:]
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return {
            "debug_enabled": self.debug_enabled,
            "breakpoints": {
                bp_id: {
                    "event_type": bp["event_type"].value if hasattr(bp["event_type"], 'value') else str(bp["event_type"]),
                    "hit_count": bp["hit_count"]
                }
                for bp_id, bp in self.event_breakpoints.items()
            },
            "captured_events_count": len(self.captured_events),
            "max_captured_events": self.max_captured_events
        }


class AuditLevel(str, Enum):
    """审计级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEventType(str, Enum):
    """审计事件类型"""
    AGENT_CREATED = "agent_created"
    AGENT_DESTROYED = "agent_destroyed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    SECURITY_EVENT = "security_event"
    CONFIGURATION_CHANGED = "configuration_changed"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM_ERROR = "system_error"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    timestamp: datetime = field(default_factory=lambda: utc_now())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    level: AuditLevel = AuditLevel.MEDIUM
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "level": self.level.value,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "request_id": self.request_id
        }

    @property
    def log_message(self) -> str:
        """生成日志消息"""
        return f"[{self.event_type.value}] {self.action} - {self.result}"


class OpenTelemetryProvider:
    """OpenTelemetry提供者"""
    
    def __init__(
        self,
        service_name: str = "ai-agent-system",
        service_version: str = "1.0.0",
        environment: str = "development",
        otlp_endpoint: Optional[str] = None,
        enable_console_export: bool = False
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint
        self.enable_console_export = enable_console_export
        
        self.tracer = None
        self.meter = None
        self.initialized = False
        
        if OTEL_AVAILABLE:
            self._initialize_telemetry()
        else:
            logger.warning("OpenTelemetry不可用")
    
    def _initialize_telemetry(self):
        """初始化遥测"""
        try:
            # 创建资源
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: self.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
                ResourceAttributes.HOST_NAME: platform.node(),
                ResourceAttributes.PROCESS_PID: os.getpid(),
                "service.instance.id": str(uuid.uuid4())
            })
            
            # 配置追踪
            trace_provider = TracerProvider(resource=resource)
            
            # 添加导出器
            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            
            if self.enable_console_export:
                console_exporter = ConsoleSpanExporter()
                trace_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            
            trace.set_tracer_provider(trace_provider)
            self.tracer = get_tracer(__name__)
            
            # 配置指标
            if self.otlp_endpoint:
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=self.otlp_endpoint),
                    export_interval_millis=10000
                )
            else:
                metric_reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=30000
                )
            
            metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(metric_provider)
            self.meter = get_meter(__name__)
            
            # 设置传播器
            set_global_textmap(B3MultiFormat())
            
            # 自动仪表化
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
            
            self.initialized = True
            logger.info("OpenTelemetry初始化完成", 
                       service_name=self.service_name,
                       environment=self.environment)
            
        except Exception as e:
            logger.error("OpenTelemetry初始化失败", error=str(e))
            self.initialized = False
    
    def create_span(
        self,
        name: str,
        kind: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """创建跨度"""
        if not self.initialized or not self.tracer:
            return None
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        return span
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        attributes: Optional[Dict[str, str]] = None
    ):
        """记录指标"""
        if not self.initialized or not self.meter:
            return
        
        try:
            # 获取或创建计量器
            instrument = getattr(self.meter, 'create_histogram', lambda n, u: None)(name, unit)
            if instrument:
                instrument.record(value, attributes or {})
        except Exception as e:
            logger.error("记录指标失败", name=name, error=str(e))


class EnterpriseAuditLogger:
    """企业级审计日志记录器"""
    
    def __init__(
        self,
        storage_backend: str = "file",
        audit_file_path: str = "/var/log/ai-agent-audit.log",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        retention_days: int = 90,
        enable_encryption: bool = True,
        enable_integrity_check: bool = True
    ):
        self.storage_backend = storage_backend
        self.audit_file_path = audit_file_path
        self.max_file_size = max_file_size
        self.retention_days = retention_days
        self.enable_encryption = enable_encryption
        self.enable_integrity_check = enable_integrity_check
        
        self.audit_events: List[AuditEvent] = []
        self.max_memory_events = 10000
        self.audit_handlers: List[Callable[[AuditEvent], None]] = []
        
        # 统计
        self.stats = {
            "events_logged": 0,
            "events_by_type": defaultdict(int),
            "events_by_level": defaultdict(int),
            "last_flush_time": None
        }
        
        logger.info("企业级审计日志记录器初始化完成", storage=storage_backend)
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        level: AuditLevel = AuditLevel.MEDIUM,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """记录审计事件"""
        audit_event = AuditEvent(
            event_type=event_type,
            action=action,
            result=result,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            resource=resource,
            details=details or {},
            level=level,
            source_ip=source_ip,
            user_agent=user_agent,
            request_id=request_id
        )
        
        # 更新统计
        self.stats["events_logged"] += 1
        self.stats["events_by_type"][event_type.value] += 1
        self.stats["events_by_level"][level.value] += 1
        
        # 内存存储
        self.audit_events.append(audit_event)
        if len(self.audit_events) > self.max_memory_events:
            self.audit_events = self.audit_events[-self.max_memory_events:]
        
        # 调用处理器
        for handler in self.audit_handlers:
            try:
                await self._call_handler(handler, audit_event)
            except Exception as e:
                logger.error("审计处理器失败", handler=handler.__name__, error=str(e))
        
        # 持久化存储
        await self._persist_audit_event(audit_event)
        
        # 记录到结构化日志
        logger.info(
            audit_event.log_message,
            event_id=audit_event.event_id,
            event_type=event_type.value,
            level=level.value,
            user_id=user_id,
            agent_id=agent_id,
            resource=resource
        )
        
        return audit_event.event_id
    
    async def _call_handler(self, handler: Callable, event: AuditEvent):
        """调用审计处理器"""
        if asyncio.iscoroutinefunction(handler):
            await handler(event)
        else:
            handler(event)
    
    async def _persist_audit_event(self, event: AuditEvent):
        """持久化审计事件"""
        try:
            if self.storage_backend == "file":
                await self._write_to_file(event)
            elif self.storage_backend == "database":
                await self._write_to_database(event)
            # 可以添加其他存储后端
            
        except Exception as e:
            logger.error("审计事件持久化失败", event_id=event.event_id, error=str(e))
    
    async def _write_to_file(self, event: AuditEvent):
        """写入文件"""
        event_data = event.to_dict()
        
        # 添加完整性检查
        if self.enable_integrity_check:
            event_data["checksum"] = self._calculate_checksum(event_data)
        
        # 序列化
        event_line = json.dumps(event_data, ensure_ascii=False) + "\n"
        
        # 加密（如果启用）
        if self.enable_encryption:
            event_line = self._encrypt_data(event_line)
        
        # 异步写入文件
        try:
            with open(self.audit_file_path, "a", encoding="utf-8") as f:
                f.write(event_line)
                f.flush()
        except Exception as e:
            logger.error("审计文件写入失败", file=self.audit_file_path, error=str(e))
    
    async def _write_to_database(self, event: AuditEvent):
        """写入数据库"""
        # TODO: 实现数据库写入逻辑
        pass
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """计算校验和"""
        # 排除checksum字段本身
        data_copy = {k: v for k, v in data.items() if k != "checksum"}
        data_str = json.dumps(data_copy, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _encrypt_data(self, data: str) -> str:
        """加密数据"""
        # 简单的Base64编码，生产环境应使用真正的加密
        import base64
        return base64.b64encode(data.encode()).decode() + "\n"
    
    def add_audit_handler(self, handler: Callable[[AuditEvent], None]):
        """添加审计处理器"""
        self.audit_handlers.append(handler)
        logger.info("添加审计处理器", handler=handler.__name__)
    
    async def search_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """搜索审计事件"""
        results = []
        
        # 从内存搜索
        for event in self.audit_events:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if agent_id and event.agent_id != agent_id:
                continue
            if level and event.level != level:
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """获取审计统计"""
        return {
            "total_events": self.stats["events_logged"],
            "events_by_type": dict(self.stats["events_by_type"]),
            "events_by_level": dict(self.stats["events_by_level"]),
            "memory_events": len(self.audit_events),
            "last_flush_time": self.stats["last_flush_time"],
            "storage_backend": self.storage_backend,
            "handlers_count": len(self.audit_handlers)
        }


class EnterpriseMonitoringManager:
    """企业级监控管理器"""
    
    def __init__(
        self,
        event_bus: EventBus,
        agent_manager: AsyncAgentManager,
        enable_otel: bool = True,
        otel_endpoint: Optional[str] = None,
        audit_storage: str = "file"
    ):
        self.event_bus = event_bus
        self.agent_manager = agent_manager
        
        # 初始化组件
        self.performance_monitor = PerformanceMonitor(event_bus)
        self.conversation_tracker = ConversationTracker(event_bus)
        self.debug_console = DebugConsole(
            event_bus, agent_manager, self.performance_monitor, self.conversation_tracker
        )
        self.dashboard = AgentDashboard(
            agent_manager, self.performance_monitor, self.conversation_tracker, event_bus
        )
        
        # OpenTelemetry提供者
        self.otel_provider = None
        if enable_otel:
            self.otel_provider = OpenTelemetryProvider(
                otlp_endpoint=otel_endpoint,
                enable_console_export=not otel_endpoint
            )
        
        # 审计日志记录器
        self.audit_logger = EnterpriseAuditLogger(storage_backend=audit_storage)
        
        # 监控事件处理器
        self.monitoring_handler = MonitoringEventHandler(
            self.performance_monitor, self.conversation_tracker
        )
        
        # 事件处理监控器
        self.event_monitor = EventProcessingMonitor()
        
        # 事件调试器
        self.event_debugger = EventDebugger(event_bus)
        
        # 健康检查
        self.health_checks: List[Callable[[], Dict[str, Any]]] = []
        self.alert_thresholds = {
            "response_time_ms": 5000,
            "error_rate": 0.1,
            "memory_usage_mb": 1024,
            "cpu_usage_percent": 80.0,
            "disk_usage_percent": 90.0
        }
        
        # 注册事件处理器
        self.event_bus.subscribe(self.monitoring_handler)
        
        logger.info("企业级监控管理器初始化完成")
    
    async def start_monitoring(self):
        """启动监控"""
        logger.info("启动企业级监控")
        
        # 记录启动审计事件
        await self.audit_logger.log_audit_event(
            AuditEventType.SYSTEM_ERROR,
            "monitoring_started",
            "success",
            details={"components": ["performance", "conversation", "audit", "telemetry"]}
        )
        
        # 启动后台任务
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._audit_flush_loop())
    
    async def stop_monitoring(self):
        """停止监控"""
        logger.info("停止企业级监控")
        
        # 记录停止审计事件
        await self.audit_logger.log_audit_event(
            AuditEventType.SYSTEM_ERROR,
            "monitoring_stopped",
            "success"
        )
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                health_status = await self._perform_health_check()
                
                # 记录健康状态指标
                if self.otel_provider:
                    self.otel_provider.record_metric(
                        "system_health_status",
                        1.0 if health_status["status"] == "healthy" else 0.0,
                        attributes={"status": health_status["status"]}
                    )
                
                # 如果健康状态异常，记录审计事件
                if health_status["status"] != "healthy":
                    await self.audit_logger.log_audit_event(
                        AuditEventType.SYSTEM_ERROR,
                        "health_check_failed",
                        "warning",
                        details=health_status,
                        level=AuditLevel.HIGH
                    )
                
            except Exception as e:
                logger.error("健康检查失败", error=str(e))
    
    async def _metrics_collection_loop(self):
        """指标收集循环"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒收集一次
                
                # 收集系统指标
                metrics = await self._collect_system_metrics()
                
                # 发送到OpenTelemetry
                if self.otel_provider:
                    for metric_name, value in metrics.items():
                        self.otel_provider.record_metric(metric_name, value)
                
                # 检查阈值
                await self._check_metric_thresholds(metrics)
                
            except Exception as e:
                logger.error("指标收集失败", error=str(e))
    
    async def _audit_flush_loop(self):
        """审计日志刷新循环"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟刷新一次
                
                # 更新统计
                self.audit_logger.stats["last_flush_time"] = utc_now().isoformat()
                
                # 这里可以添加日志轮转、压缩等逻辑
                
            except Exception as e:
                logger.error("审计日志刷新失败", error=str(e))
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": utc_now().isoformat(),
            "checks": {}
        }
        
        # 执行各种健康检查
        checks = [
            ("agent_manager", self._check_agent_manager_health),
            ("event_bus", self._check_event_bus_health),
            ("performance", self._check_performance_health),
            ("storage", self._check_storage_health)
        ]
        
        overall_status = "healthy"
        
        for check_name, check_func in checks:
            try:
                check_result = await check_func()
                health_status["checks"][check_name] = check_result
                
                if check_result["status"] != "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                health_status["checks"][check_name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        health_status["status"] = overall_status
        return health_status
    
    async def _check_agent_manager_health(self) -> Dict[str, Any]:
        """检查智能体管理器健康状态"""
        try:
            stats = self.agent_manager.get_manager_stats()
            
            # 检查任务队列积压
            pending_tasks = stats.get("tasks", {}).get("pending", 0)
            if pending_tasks > 100:
                return {"status": "degraded", "reason": "高任务队列积压", "pending_tasks": pending_tasks}
            
            return {"status": "healthy", "stats": stats}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_event_bus_health(self) -> Dict[str, Any]:
        """检查事件总线健康状态"""
        try:
            stats = self.event_bus.get_stats()
            
            # 检查事件积压
            pending_events = stats.get("pending_events", 0)
            if pending_events > 1000:
                return {"status": "degraded", "reason": "高事件积压", "pending_events": pending_events}
            
            return {"status": "healthy", "stats": stats}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态"""
        try:
            metrics = self.performance_monitor.get_all_metrics_summary(5)
            
            # 检查响应时间
            response_time = metrics.get("summaries", {}).get("response_time", {}).get("avg", 0)
            if response_time > self.alert_thresholds["response_time_ms"]:
                return {"status": "degraded", "reason": "高响应时间", "avg_response_time": response_time}
            
            return {"status": "healthy", "metrics": metrics}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_storage_health(self) -> Dict[str, Any]:
        """检查存储健康状态"""
        try:
            # 检查磁盘使用率
            import shutil
            disk_usage = shutil.disk_usage("/")
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
                return {"status": "degraded", "reason": "高磁盘使用率", "disk_usage_percent": disk_usage_percent}
            
            return {"status": "healthy", "disk_usage_percent": disk_usage_percent}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        metrics = {}
        
        try:
            import psutil
            
            # CPU使用率
            metrics["system_cpu_percent"] = psutil.cpu_percent()
            
            # 内存使用率
            memory = psutil.virtual_memory()
            metrics["system_memory_percent"] = memory.percent
            metrics["system_memory_used_mb"] = memory.used / (1024 * 1024)
            
            # 磁盘使用率
            disk = psutil.disk_usage("/")
            metrics["system_disk_percent"] = (disk.used / disk.total) * 100
            
            # 网络IO
            network = psutil.net_io_counters()
            metrics["system_network_bytes_sent"] = network.bytes_sent
            metrics["system_network_bytes_recv"] = network.bytes_recv
            
            # 进程信息
            process = psutil.Process()
            metrics["process_cpu_percent"] = process.cpu_percent()
            metrics["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            metrics["process_threads"] = process.num_threads()
            
        except Exception as e:
            logger.error("系统指标收集失败", error=str(e))
        
        return metrics
    
    async def _check_metric_thresholds(self, metrics: Dict[str, float]):
        """检查指标阈值"""
        for metric_name, value in metrics.items():
            threshold_key = metric_name.replace("system_", "").replace("process_", "")
            threshold = self.alert_thresholds.get(threshold_key)
            
            if threshold and value > threshold:
                await self.audit_logger.log_audit_event(
                    AuditEventType.SYSTEM_ERROR,
                    "metric_threshold_exceeded",
                    "warning",
                    details={
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold
                    },
                    level=AuditLevel.HIGH
                )
    
    def add_health_check(self, check_func: Callable[[], Dict[str, Any]]):
        """添加自定义健康检查"""
        self.health_checks.append(check_func)
    
    async def get_monitoring_overview(self) -> Dict[str, Any]:
        """获取监控总览"""
        return {
            "timestamp": utc_now().isoformat(),
            "dashboard": await self.dashboard.get_dashboard_data(),
            "health": await self._perform_health_check(),
            "audit_stats": self.audit_logger.get_audit_statistics(),
            "otel_enabled": self.otel_provider is not None and self.otel_provider.initialized,
            "event_processing": await self.event_monitor.get_processing_stats(),
            "debug_info": self.event_debugger.get_debug_info()
        }