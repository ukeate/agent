"""
工作流执行监控系统
提供实时监控、性能分析和告警功能
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import redis.asyncio as redis
from pydantic import BaseModel

from models.schemas.workflow import WorkflowExecution, WorkflowStepStatus
from src.ai.workflow.scheduler import ScheduledTask, TaskStatus, WorkflowScheduler
from src.core.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MonitoringAlert:
    """监控告警"""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    execution_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class ExecutionMetrics:
    """执行指标"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    completed_steps: int
    failed_steps: int
    running_steps: int
    pending_steps: int
    average_step_duration: float  # 秒
    total_duration: Optional[float]  # 秒
    success_rate: float
    throughput: float  # 每分钟完成步骤数
    resource_usage: Dict[str, float]
    bottlenecks: List[str]
    critical_path_duration: float
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    active_executions: int
    total_queued_tasks: int
    active_workers: int
    average_queue_wait_time: float  # 秒
    system_throughput: float  # 每分钟处理任务数
    error_rate: float
    resource_utilization: Dict[str, float]
    queue_depths: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.execution_history: Dict[str, List[ExecutionMetrics]] = defaultdict(list)
        self.system_history: deque = deque(maxlen=1000)  # 保留最近1000个数据点
        self.bottleneck_patterns: Dict[str, int] = defaultdict(int)
    
    def analyze_execution_performance(self, metrics: ExecutionMetrics) -> Dict[str, Any]:
        """分析执行性能"""
        analysis = {
            "performance_score": 0.0,
            "efficiency_score": 0.0,
            "reliability_score": 0.0,
            "bottlenecks": [],
            "recommendations": [],
            "trends": {}
        }
        
        try:
            # 性能评分 (0-100)
            if metrics.total_duration and metrics.critical_path_duration > 0:
                efficiency = min(metrics.critical_path_duration / metrics.total_duration, 1.0)
                analysis["efficiency_score"] = efficiency * 100
            
            # 可靠性评分
            analysis["reliability_score"] = metrics.success_rate * 100
            
            # 综合性能评分
            analysis["performance_score"] = (
                analysis["efficiency_score"] * 0.4 + 
                analysis["reliability_score"] * 0.4 + 
                min(metrics.throughput * 10, 100) * 0.2
            )
            
            # 瓶颈分析
            if metrics.average_step_duration > 60:  # 超过1分钟
                analysis["bottlenecks"].append("步骤执行时间过长")
            
            if metrics.pending_steps > metrics.running_steps * 2:
                analysis["bottlenecks"].append("任务排队时间过长")
            
            if metrics.failed_steps > metrics.total_steps * 0.1:
                analysis["bottlenecks"].append("失败率过高")
            
            # 优化建议
            if analysis["efficiency_score"] < 60:
                analysis["recommendations"].append("考虑增加并行度或优化任务分解")
            
            if analysis["reliability_score"] < 80:
                analysis["recommendations"].append("检查任务配置和依赖关系")
            
            if metrics.throughput < 1.0:
                analysis["recommendations"].append("考虑增加工作器数量或优化资源配置")
            
            # 趋势分析
            if metrics.execution_id in self.execution_history:
                historical_metrics = self.execution_history[metrics.execution_id]
                if len(historical_metrics) > 1:
                    prev_metrics = historical_metrics[-2]
                    
                    analysis["trends"] = {
                        "duration_trend": self._calculate_trend(
                            prev_metrics.total_duration or 0, 
                            metrics.total_duration or 0
                        ),
                        "success_rate_trend": self._calculate_trend(
                            prev_metrics.success_rate, 
                            metrics.success_rate
                        ),
                        "throughput_trend": self._calculate_trend(
                            prev_metrics.throughput, 
                            metrics.throughput
                        )
                    }
            
            # 记录到历史
            self.execution_history[metrics.execution_id].append(metrics)
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
        
        return analysis
    
    def analyze_system_performance(self, current_metrics: SystemMetrics) -> Dict[str, Any]:
        """分析系统性能"""
        analysis = {
            "system_health_score": 0.0,
            "load_level": "normal",
            "capacity_utilization": 0.0,
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # 系统健康评分
            health_factors = []
            
            # 错误率评分 (错误率越低越好)
            error_factor = max(0, 100 - current_metrics.error_rate * 100)
            health_factors.append(error_factor)
            
            # 队列等待时间评分
            wait_time_factor = max(0, 100 - min(current_metrics.average_queue_wait_time / 60, 1) * 100)
            health_factors.append(wait_time_factor)
            
            # 资源利用率评分 (70-80%最优)
            avg_utilization = sum(current_metrics.resource_utilization.values()) / max(len(current_metrics.resource_utilization), 1)
            if 0.7 <= avg_utilization <= 0.8:
                utilization_factor = 100
            elif avg_utilization < 0.7:
                utilization_factor = avg_utilization / 0.7 * 100
            else:
                utilization_factor = max(0, 100 - (avg_utilization - 0.8) * 500)
            health_factors.append(utilization_factor)
            
            analysis["system_health_score"] = sum(health_factors) / len(health_factors)
            analysis["capacity_utilization"] = avg_utilization * 100
            
            # 负载水平评估
            if current_metrics.total_queued_tasks > 100:
                analysis["load_level"] = "high"
            elif current_metrics.total_queued_tasks > 50:
                analysis["load_level"] = "medium"
            else:
                analysis["load_level"] = "low"
            
            # 告警检查
            if current_metrics.error_rate > 0.1:  # 错误率超过10%
                analysis["alerts"].append({
                    "level": "warning",
                    "message": f"错误率过高: {current_metrics.error_rate:.2%}"
                })
            
            if current_metrics.average_queue_wait_time > 300:  # 等待时间超过5分钟
                analysis["alerts"].append({
                    "level": "warning", 
                    "message": f"队列等待时间过长: {current_metrics.average_queue_wait_time:.1f}秒"
                })
            
            if avg_utilization > 0.9:  # 资源利用率超过90%
                analysis["alerts"].append({
                    "level": "critical",
                    "message": f"资源利用率过高: {avg_utilization:.1%}"
                })
            
            # 优化建议
            if analysis["load_level"] == "high":
                analysis["recommendations"].append("考虑增加工作器数量")
            
            if current_metrics.average_queue_wait_time > 180:
                analysis["recommendations"].append("优化任务调度策略")
            
            if avg_utilization < 0.3:
                analysis["recommendations"].append("考虑减少资源配置以降低成本")
            
            # 记录到历史
            self.system_history.append(current_metrics)
            
        except Exception as e:
            logger.error(f"系统性能分析失败: {e}")
        
        return analysis
    
    def _calculate_trend(self, previous_value: float, current_value: float) -> str:
        """计算趋势"""
        if previous_value == 0:
            return "stable"
        
        change_rate = (current_value - previous_value) / previous_value
        
        if change_rate > 0.1:
            return "improving"
        elif change_rate < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_performance_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取性能摘要"""
        cutoff_time = utc_now() - timedelta(hours=time_range_hours)
        
        summary = {
            "time_range_hours": time_range_hours,
            "total_executions": 0,
            "average_performance_score": 0.0,
            "average_success_rate": 0.0,
            "common_bottlenecks": [],
            "performance_trends": {},
            "system_stability": "stable"
        }
        
        try:
            all_metrics = []
            for execution_metrics_list in self.execution_history.values():
                for metrics in execution_metrics_list:
                    if metrics.last_updated >= cutoff_time:
                        all_metrics.append(metrics)
            
            if all_metrics:
                summary["total_executions"] = len(all_metrics)
                summary["average_success_rate"] = sum(m.success_rate for m in all_metrics) / len(all_metrics)
                
                # 常见瓶颈
                bottleneck_counts = defaultdict(int)
                for metrics in all_metrics:
                    for bottleneck in metrics.bottlenecks:
                        bottleneck_counts[bottleneck] += 1
                
                summary["common_bottlenecks"] = sorted(
                    bottleneck_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            
            # 系统稳定性
            recent_system_metrics = [m for m in self.system_history if m.timestamp >= cutoff_time]
            if recent_system_metrics:
                error_rates = [m.error_rate for m in recent_system_metrics]
                avg_error_rate = sum(error_rates) / len(error_rates)
                
                if avg_error_rate > 0.1:
                    summary["system_stability"] = "unstable"
                elif avg_error_rate > 0.05:
                    summary["system_stability"] = "moderate"
                else:
                    summary["system_stability"] = "stable"
        
        except Exception as e:
            logger.error(f"生成性能摘要失败: {e}")
        
        return summary


class ExecutionMonitor:
    """执行监控器"""
    
    def __init__(self, redis_client: redis.Redis, scheduler: WorkflowScheduler):
        self.redis = redis_client
        self.scheduler = scheduler
        self.performance_analyzer = PerformanceAnalyzer()
        self.alert_handlers: List[Callable[[MonitoringAlert], None]] = []
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.monitoring_tasks: Set[asyncio.Task] = set()
        self.running = False
        
        # 监控配置
        self.monitoring_interval = 30  # 秒
        self.alert_thresholds = {
            "max_execution_duration": 3600,  # 1小时
            "max_step_duration": 600,        # 10分钟  
            "max_queue_wait_time": 300,      # 5分钟
            "min_success_rate": 0.8,         # 80%
            "max_error_rate": 0.1,           # 10%
        }
    
    async def start_monitoring(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        logger.info("执行监控器启动")
        
        # 启动监控任务
        self.monitoring_tasks.add(
            asyncio.create_task(self._execution_monitoring_loop())
        )
        self.monitoring_tasks.add(
            asyncio.create_task(self._system_monitoring_loop())
        )
        self.monitoring_tasks.add(
            asyncio.create_task(self._alert_processing_loop())
        )
    
    async def stop_monitoring(self):
        """停止监控"""
        self.running = False
        
        # 取消所有监控任务
        for task in self.monitoring_tasks:
            task.cancel()
        
        # 等待任务完成
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("执行监控器停止")
    
    def add_alert_handler(self, handler: Callable[[MonitoringAlert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    async def track_execution(self, execution: WorkflowExecution):
        """跟踪执行"""
        self.active_executions[execution.id] = execution
        logger.info(f"开始跟踪执行: {execution.id}")
    
    async def untrack_execution(self, execution_id: str):
        """停止跟踪执行"""
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
            logger.info(f"停止跟踪执行: {execution_id}")
    
    async def _execution_monitoring_loop(self):
        """执行监控循环"""
        while self.running:
            try:
                for execution_id, execution in self.active_executions.items():
                    await self._monitor_execution(execution)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"执行监控循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _system_monitoring_loop(self):
        """系统监控循环"""
        while self.running:
            try:
                await self._monitor_system()
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"系统监控循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _alert_processing_loop(self):
        """告警处理循环"""
        while self.running:
            try:
                # 这里可以处理告警队列
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"告警处理循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_execution(self, execution: WorkflowExecution):
        """监控单个执行"""
        try:
            # 收集执行指标
            metrics = await self._collect_execution_metrics(execution)
            
            # 性能分析
            analysis = self.performance_analyzer.analyze_execution_performance(metrics)
            
            # 检查告警条件
            await self._check_execution_alerts(execution, metrics, analysis)
            
            # 存储指标数据
            await self._store_metrics(execution.id, metrics, analysis)
            
        except Exception as e:
            logger.error(f"监控执行失败: {execution.id}, 错误: {e}")
    
    async def _monitor_system(self):
        """监控系统"""
        try:
            # 收集系统指标
            system_metrics = await self._collect_system_metrics()
            
            # 性能分析
            analysis = self.performance_analyzer.analyze_system_performance(system_metrics)
            
            # 检查系统告警
            await self._check_system_alerts(system_metrics, analysis)
            
            # 存储系统指标
            await self._store_system_metrics(system_metrics, analysis)
            
        except Exception as e:
            logger.error(f"系统监控失败: {e}")
    
    async def _collect_execution_metrics(self, execution: WorkflowExecution) -> ExecutionMetrics:
        """收集执行指标"""
        now = utc_now()
        
        # 统计步骤状态
        status_counts = defaultdict(int)
        step_durations = []
        
        for step_exec in execution.step_executions:
            status_counts[step_exec.status] += 1
            
            if step_exec.duration_ms:
                step_durations.append(step_exec.duration_ms / 1000.0)
        
        # 计算指标
        total_steps = len(execution.step_executions)
        completed_steps = status_counts[WorkflowStepStatus.COMPLETED]
        failed_steps = status_counts[WorkflowStepStatus.FAILED]
        running_steps = status_counts[WorkflowStepStatus.RUNNING]
        pending_steps = status_counts[WorkflowStepStatus.PENDING]
        
        average_step_duration = sum(step_durations) / len(step_durations) if step_durations else 0.0
        success_rate = completed_steps / total_steps if total_steps > 0 else 0.0
        
        total_duration = None
        if execution.started_at:
            if execution.completed_at:
                total_duration = (execution.completed_at - execution.started_at).total_seconds()
            else:
                total_duration = (now - execution.started_at).total_seconds()
        
        # 计算吞吐量
        throughput = 0.0
        if total_duration and total_duration > 0:
            throughput = (completed_steps / total_duration) * 60  # 每分钟
        
        # 分析瓶颈
        bottlenecks = []
        if average_step_duration > 300:  # 5分钟
            bottlenecks.append("步骤执行缓慢")
        if failed_steps > total_steps * 0.2:
            bottlenecks.append("高失败率")
        if pending_steps > running_steps * 3:
            bottlenecks.append("队列积压")
        
        return ExecutionMetrics(
            execution_id=execution.id,
            start_time=execution.started_at or execution.created_at,
            end_time=execution.completed_at,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            running_steps=running_steps,
            pending_steps=pending_steps,
            average_step_duration=average_step_duration,
            total_duration=total_duration,
            success_rate=success_rate,
            throughput=throughput,
            resource_usage={},  # TODO: 实现资源使用统计
            bottlenecks=bottlenecks,
            critical_path_duration=total_duration or 0.0,  # 简化实现
            last_updated=now
        )
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        now = utc_now()
        
        # 获取调度器统计
        scheduler_stats = await self.scheduler.get_scheduler_stats()
        queue_stats = scheduler_stats.get('queue_stats', {})
        
        # 计算系统指标
        active_executions = len(self.active_executions)
        total_queued_tasks = queue_stats.get('total_queued', 0)
        active_workers = scheduler_stats.get('active_workers', 0)
        
        # 计算错误率
        total_tasks = queue_stats.get('total_tasks', 1)
        failed_tasks = queue_stats.get('failed', 0)
        error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # 简化的系统指标
        return SystemMetrics(
            timestamp=now,
            active_executions=active_executions,
            total_queued_tasks=total_queued_tasks,
            active_workers=active_workers,
            average_queue_wait_time=30.0,  # TODO: 实现真实的等待时间计算
            system_throughput=10.0,        # TODO: 实现真实的系统吞吐量
            error_rate=error_rate,
            resource_utilization={"cpu": 0.5, "memory": 0.3},  # TODO: 实现真实的资源监控
            queue_depths={
                "high": queue_stats.get('high_priority', 0),
                "normal": queue_stats.get('normal_priority', 0),
                "low": queue_stats.get('low_priority', 0)
            }
        )
    
    async def _check_execution_alerts(self, execution: WorkflowExecution, metrics: ExecutionMetrics, analysis: Dict[str, Any]):
        """检查执行告警"""
        alerts = []
        
        # 检查执行时间过长
        if metrics.total_duration and metrics.total_duration > self.alert_thresholds["max_execution_duration"]:
            alerts.append(MonitoringAlert(
                id=f"exec_duration_{execution.id}",
                level=AlertLevel.WARNING,
                title="执行时间过长",
                message=f"执行时间已超过 {metrics.total_duration/60:.1f} 分钟",
                source="execution_monitor",
                timestamp=utc_now(),
                execution_id=execution.id
            ))
        
        # 检查成功率
        if metrics.success_rate < self.alert_thresholds["min_success_rate"]:
            alerts.append(MonitoringAlert(
                id=f"success_rate_{execution.id}",
                level=AlertLevel.ERROR,
                title="成功率过低",
                message=f"成功率仅为 {metrics.success_rate:.1%}",
                source="execution_monitor",
                timestamp=utc_now(),
                execution_id=execution.id
            ))
        
        # 检查性能评分
        if analysis.get("performance_score", 0) < 50:
            alerts.append(MonitoringAlert(
                id=f"performance_{execution.id}",
                level=AlertLevel.WARNING,
                title="性能评分低",
                message=f"性能评分仅为 {analysis['performance_score']:.1f}",
                source="execution_monitor",
                timestamp=utc_now(),
                execution_id=execution.id
            ))
        
        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _check_system_alerts(self, metrics: SystemMetrics, analysis: Dict[str, Any]):
        """检查系统告警"""
        alerts = []
        
        # 检查错误率
        if metrics.error_rate > self.alert_thresholds["max_error_rate"]:
            alerts.append(MonitoringAlert(
                id="system_error_rate",
                level=AlertLevel.ERROR,
                title="系统错误率过高",
                message=f"错误率达到 {metrics.error_rate:.1%}",
                source="system_monitor",
                timestamp=utc_now()
            ))
        
        # 检查队列积压
        if metrics.total_queued_tasks > 100:
            alerts.append(MonitoringAlert(
                id="queue_backlog",
                level=AlertLevel.WARNING,
                title="队列积压",
                message=f"队列中有 {metrics.total_queued_tasks} 个任务待处理",
                source="system_monitor",
                timestamp=utc_now()
            ))
        
        # 检查工作器数量
        if metrics.active_workers == 0 and metrics.total_queued_tasks > 0:
            alerts.append(MonitoringAlert(
                id="no_workers",
                level=AlertLevel.CRITICAL,
                title="无可用工作器",
                message="没有活跃的工作器处理任务",
                source="system_monitor",
                timestamp=utc_now()
            ))
        
        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: MonitoringAlert):
        """发送告警"""
        try:
            # 存储告警到Redis
            alert_key = f"alerts:{alert.id}"
            await self.redis.hset(alert_key, mapping=alert.to_dict())
            await self.redis.expire(alert_key, 86400)  # 24小时过期
            
            # 调用告警处理器
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"告警处理器执行失败: {e}")
            
            logger.info(f"发送告警: {alert.title} - {alert.message}")
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
    
    async def _store_metrics(self, execution_id: str, metrics: ExecutionMetrics, analysis: Dict[str, Any]):
        """存储指标数据"""
        try:
            # 存储到Redis
            metrics_key = f"metrics:execution:{execution_id}"
            data = {
                "metrics": json.dumps(metrics.to_dict()),
                "analysis": json.dumps(analysis),
                "timestamp": utc_now().isoformat()
            }
            
            await self.redis.hset(metrics_key, mapping=data)
            await self.redis.expire(metrics_key, 86400 * 7)  # 7天过期
            
        except Exception as e:
            logger.error(f"存储执行指标失败: {execution_id}, 错误: {e}")
    
    async def _store_system_metrics(self, metrics: SystemMetrics, analysis: Dict[str, Any]):
        """存储系统指标"""
        try:
            # 存储到Redis时序数据
            timestamp = int(metrics.timestamp.timestamp())
            metrics_key = f"metrics:system:{timestamp}"
            
            data = {
                "metrics": json.dumps(metrics.to_dict()),
                "analysis": json.dumps(analysis)
            }
            
            await self.redis.hset(metrics_key, mapping=data)
            await self.redis.expire(metrics_key, 86400 * 7)  # 7天过期
            
        except Exception as e:
            logger.error(f"存储系统指标失败: {e}")
    
    async def get_execution_metrics(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行指标"""
        try:
            metrics_key = f"metrics:execution:{execution_id}"
            data = await self.redis.hgetall(metrics_key)
            
            if data:
                return {
                    "metrics": json.loads(data.get(b"metrics", b"{}")),
                    "analysis": json.loads(data.get(b"analysis", b"{}")),
                    "timestamp": data.get(b"timestamp", b"").decode()
                }
        except Exception as e:
            logger.error(f"获取执行指标失败: {execution_id}, 错误: {e}")
        
        return None
    
    async def get_system_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取系统指标历史"""
        try:
            cutoff_time = utc_now() - timedelta(hours=hours)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            # 扫描时间范围内的指标
            pattern = "metrics:system:*"
            metrics_history = []
            
            async for key in self.redis.scan_iter(match=pattern):
                try:
                    timestamp = int(key.decode().split(":")[-1])
                    if timestamp >= cutoff_timestamp:
                        data = await self.redis.hgetall(key)
                        if data:
                            metrics_history.append({
                                "timestamp": timestamp,
                                "metrics": json.loads(data.get(b"metrics", b"{}")),
                                "analysis": json.loads(data.get(b"analysis", b"{}"))
                            })
                except (ValueError, json.JSONDecodeError):
                    continue
            
            # 按时间排序
            metrics_history.sort(key=lambda x: x["timestamp"])
            return metrics_history
            
        except Exception as e:
            logger.error(f"获取系统指标历史失败: {e}")
            return []


# 简单的告警处理器示例
async def console_alert_handler(alert: MonitoringAlert):
    """控制台告警处理器"""
    level_colors = {
        AlertLevel.INFO: "\033[32m",      # 绿色
        AlertLevel.WARNING: "\033[33m",   # 黄色  
        AlertLevel.ERROR: "\033[31m",     # 红色
        AlertLevel.CRITICAL: "\033[35m"   # 紫色
    }
    
    color = level_colors.get(alert.level, "")
    reset_color = "\033[0m"
    
    print(f"{color}[{alert.level.upper()}] {alert.title}{reset_color}")
    print(f"  消息: {alert.message}")
    print(f"  时间: {alert.timestamp}")
    if alert.execution_id:
        print(f"  执行ID: {alert.execution_id}")
    print("-" * 50)