"""
实时指标API端点
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Request, Depends
from pydantic import Field, field_validator, ConfigDict
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.ai.cluster import MetricsCollector
from src.services.realtime_metrics_service import (
    get_realtime_metrics_service,
    MetricDefinition,
    MetricCategory,
    MetricType,
    AggregationType,
    MetricDefinition,
    MetricCategory,
    MetricType,
    AggregationType,
    TimeWindow
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/realtime-metrics", tags=["实时指标"])

async def get_metrics_collector(request: Request) -> MetricsCollector:
    metrics_collector = getattr(request.app.state, "metrics_collector", None)
    if metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    return metrics_collector

def _pick_metric_value(summary: Dict[str, Any], name: str) -> Optional[float]:
    metric = summary.get(name)
    if not metric:
        return None
    latest = metric.get("latest")
    return latest if latest is not None else metric.get("average")

def _format_timestamp() -> str:
    return utc_now().isoformat()

def _metric_unit(name: str) -> str:
    unit_map = {
        "cpu_usage_percent": "%",
        "memory_usage_percent": "%",
        "storage_usage_percent": "%",
        "gpu_usage_percent": "%",
        "network_io_mbps": "Mbps",
        "avg_response_time": "ms",
        "error_rate": "%",
        "total_requests": "req",
        "failed_requests": "req",
        "active_tasks": "tasks",
    }
    return unit_map.get(name, "")

def _compute_series_rate(series: Any, duration_seconds: float = 300) -> Optional[float]:
    if not series or not getattr(series, "points", None):
        return None
    cutoff = utc_now().timestamp() - duration_seconds
    points = [p for p in series.points if p.timestamp >= cutoff]
    if len(points) < 2:
        return None
    points.sort(key=lambda p: p.timestamp)
    elapsed = points[-1].timestamp - points[0].timestamp
    if elapsed <= 0:
        return None
    delta = points[-1].value - points[0].value
    if delta < 0:
        return None
    return delta / elapsed

# 请求模型
class MetricDefinitionRequest(ApiBaseModel):
    """指标定义请求"""
    name: str = Field(..., description="指标名称")
    display_name: str = Field(..., description="显示名称")
    metric_type: str = Field(..., description="指标类型")
    category: str = Field(..., description="指标类别")
    aggregation: str = Field(..., description="聚合类型")
    unit: str = Field("", description="单位")
    description: str = Field("", description="描述")
    formula: Optional[str] = Field(None, description="计算公式")
    numerator_event: Optional[str] = Field(None, description="分子事件")
    denominator_event: Optional[str] = Field(None, description="分母事件")
    threshold_lower: Optional[float] = Field(None, description="下限阈值")
    threshold_upper: Optional[float] = Field(None, description="上限阈值")
    
    @field_validator('metric_type')
    def validate_metric_type(cls, v):
        allowed = ["conversion", "continuous", "count", "ratio"]
        if v not in allowed:
            raise ValueError(f"Metric type must be one of {allowed}")
        return v

    @field_validator('category')
    def validate_category(cls, v):
        allowed = ["primary", "secondary", "guardrail", "diagnostic"]
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v

class MetricsCalculationRequest(ApiBaseModel):
    """指标计算请求"""
    experiment_id: str = Field(..., description="实验ID")
    time_window: TimeWindow = Field(TimeWindow.CUMULATIVE, description="时间窗口")
    metrics: Optional[List[str]] = Field(None, description="指定计算的指标")

class GroupComparisonRequest(ApiBaseModel):
    """分组比较请求"""
    experiment_id: str = Field(..., description="实验ID")
    control_group: str = Field(..., description="对照组ID")
    treatment_group: str = Field(..., description="实验组ID")
    metrics: Optional[List[str]] = Field(None, description="指定比较的指标")

class MetricTrendsRequest(ApiBaseModel):
    """指标趋势请求"""
    experiment_id: str = Field(..., description="实验ID")
    metric_name: str = Field(..., description="指标名称")
    granularity: TimeWindow = Field(TimeWindow.HOURLY, description="时间粒度")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")

# 响应模型
class MetricSnapshotResponse(ApiBaseModel):
    """指标快照响应"""
    metric_name: str
    value: float
    sample_size: int
    confidence_interval: Optional[List[float]]
    timestamp: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metric_name": "conversion_rate",
                "value": 0.15,
                "sample_size": 1000,
                "confidence_interval": [0.13, 0.17],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )

class GroupMetricsResponse(ApiBaseModel):
    """分组指标响应"""
    group_id: str
    group_name: str
    metrics: Dict[str, MetricSnapshotResponse]
    user_count: int
    event_count: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "group_id": "control",
                "group_name": "Control Group",
                "metrics": {
                    "conversion_rate": {
                        "metric_name": "conversion_rate",
                        "value": 0.15,
                        "sample_size": 500
                    }
                },
                "user_count": 500,
                "event_count": 1500
            }
        }
    )

class MetricComparisonResponse(ApiBaseModel):
    """指标比较响应"""
    metric_name: str
    control_value: float
    treatment_value: float
    absolute_difference: float
    relative_difference: float
    p_value: Optional[float]
    is_significant: bool
    confidence_interval: Optional[List[float]]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metric_name": "conversion_rate",
                "control_value": 0.15,
                "treatment_value": 0.18,
                "absolute_difference": 0.03,
                "relative_difference": 20.0,
                "p_value": 0.023,
                "is_significant": True,
                "confidence_interval": [0.01, 0.05]
            }
        }
    )

class RealtimeMetricsResponse(ApiBaseModel):
    """实时指标响应"""
    experiment_id: str
    groups: Dict[str, GroupMetricsResponse]
    timestamp: str
    time_window: str
    message: str = Field(default="Metrics calculated successfully")

class MetricTrendsResponse(ApiBaseModel):
    """指标趋势响应"""
    metric_name: str
    trends: List[MetricSnapshotResponse]
    granularity: str
    message: str = Field(default="Trends retrieved successfully")

# API端点
@router.get("/overview")
async def get_realtime_overview(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    """获取实时指标总览（真实采集）"""
    summary = await metrics_collector.get_metrics_summary()
    timestamp = _format_timestamp()
    metrics = []
    for name in sorted(summary.keys()):
        value = _pick_metric_value(summary, name)
        if value is None:
            continue
        metrics.append(
            {
                "name": name,
                "value": value,
                "unit": _metric_unit(name),
                "timestamp": timestamp,
            }
        )
    return {"metrics": metrics}

@router.get("/services")
async def get_service_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    """获取服务级别指标（基于智能体实时指标）"""
    topology = await metrics_collector.cluster_manager.get_cluster_topology()
    services = []
    for agent_id, agent in topology.agents.items():
        summary = await metrics_collector.get_metrics_summary(agent_id=agent_id)
        payload: Dict[str, Any] = {"service": agent.name or agent.agent_id}

        latency = _pick_metric_value(summary, "avg_response_time")
        if latency is not None:
            payload["latency_ms"] = latency

        error_rate = _pick_metric_value(summary, "error_rate")
        if error_rate is not None:
            payload["error_rate"] = error_rate

        series = metrics_collector.metric_series.get("total_requests", {}).get(agent_id)
        throughput = _compute_series_rate(series)
        if throughput is not None:
            payload["throughput"] = throughput

        services.append(payload)
    return {"services": services}

@router.post("/register-metric")
async def register_metric_definition(request: MetricDefinitionRequest):
    """注册新的指标定义"""
    try:
        service = await get_realtime_metrics_service()
        
        # 创建指标定义
        metric_def = MetricDefinition(
            name=request.name,
            display_name=request.display_name,
            metric_type=MetricType(request.metric_type),
            category=MetricCategory(request.category),
            aggregation=AggregationType(request.aggregation),
            unit=request.unit,
            description=request.description,
            formula=request.formula,
            numerator_event=request.numerator_event,
            denominator_event=request.denominator_event,
            threshold_lower=request.threshold_lower,
            threshold_upper=request.threshold_upper
        )
        
        service.register_metric(metric_def)
        
        return {
            "status": "success",
            "metric": metric_def.to_dict(),
            "message": f"Metric '{request.name}' registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register metric: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to register metric: {str(e)}")

@router.post("/calculate", response_model=RealtimeMetricsResponse)
async def calculate_experiment_metrics(request: MetricsCalculationRequest):
    """计算实验指标"""
    try:
        service = await get_realtime_metrics_service()
        
        # 计算指标
        group_metrics = await service.calculate_metrics(
            experiment_id=request.experiment_id,
            time_window=request.time_window
        )
        
        # 转换响应格式
        groups_response = {}
        for group_id, metrics in group_metrics.items():
            metrics_dict = {}
            for metric_name, snapshot in metrics.metrics.items():
                if not request.metrics or metric_name in request.metrics:
                    metrics_dict[metric_name] = MetricSnapshotResponse(
                        metric_name=snapshot.metric_name,
                        value=snapshot.value,
                        sample_size=snapshot.sample_size,
                        confidence_interval=list(snapshot.confidence_interval) if snapshot.confidence_interval else None,
                        timestamp=snapshot.timestamp.isoformat()
                    )
            
            groups_response[group_id] = GroupMetricsResponse(
                group_id=group_id,
                group_name=metrics.group_name,
                metrics=metrics_dict,
                user_count=metrics.user_count,
                event_count=metrics.event_count
            )
        
        return RealtimeMetricsResponse(
            experiment_id=request.experiment_id,
            groups=groups_response,
            timestamp=utc_now().isoformat(),
            time_window=request.time_window.value,
            message=f"Calculated {len(groups_response)} groups with metrics"
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")

@router.post("/compare-groups")
async def compare_experiment_groups(request: GroupComparisonRequest):
    """比较实验组"""
    try:
        service = await get_realtime_metrics_service()
        
        # 执行比较
        comparisons = await service.compare_groups(
            experiment_id=request.experiment_id,
            control_group=request.control_group,
            treatment_group=request.treatment_group
        )
        
        # 过滤指定的指标
        if request.metrics:
            comparisons = {
                k: v for k, v in comparisons.items() 
                if k in request.metrics
            }
        
        # 转换响应格式
        comparison_results = {}
        for metric_name, comparison in comparisons.items():
            comparison_results[metric_name] = MetricComparisonResponse(
                metric_name=comparison.metric_name,
                control_value=comparison.control_value,
                treatment_value=comparison.treatment_value,
                absolute_difference=comparison.absolute_difference,
                relative_difference=comparison.relative_difference,
                p_value=comparison.p_value,
                is_significant=comparison.is_significant,
                confidence_interval=list(comparison.confidence_interval) if comparison.confidence_interval else None
            )
        
        # 生成摘要
        significant_metrics = [
            name for name, comp in comparisons.items() 
            if comp.is_significant
        ]
        
        return {
            "experiment_id": request.experiment_id,
            "control_group": request.control_group,
            "treatment_group": request.treatment_group,
            "comparisons": comparison_results,
            "summary": {
                "total_metrics": len(comparison_results),
                "significant_metrics": len(significant_metrics),
                "significant_metric_names": significant_metrics
            },
            "message": f"Compared {len(comparison_results)} metrics between groups"
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compare groups: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compare groups: {str(e)}")

@router.post("/trends", response_model=MetricTrendsResponse)
async def get_metric_trends(request: MetricTrendsRequest):
    """获取指标趋势"""
    try:
        service = await get_realtime_metrics_service()
        
        # 获取趋势数据
        trends = await service.get_metric_trends(
            experiment_id=request.experiment_id,
            metric_name=request.metric_name,
            granularity=request.granularity
        )
        
        # 过滤时间范围
        if request.start_time or request.end_time:
            filtered_trends = []
            for trend in trends:
                if request.start_time and trend.timestamp < request.start_time:
                    continue
                if request.end_time and trend.timestamp > request.end_time:
                    continue
                filtered_trends.append(trend)
            trends = filtered_trends
        
        # 转换响应格式
        trends_response = [
            MetricSnapshotResponse(
                metric_name=trend.metric_name,
                value=trend.value,
                sample_size=trend.sample_size,
                confidence_interval=list(trend.confidence_interval) if trend.confidence_interval else None,
                timestamp=trend.timestamp.isoformat()
            )
            for trend in trends
        ]
        
        return MetricTrendsResponse(
            metric_name=request.metric_name,
            trends=trends_response,
            granularity=request.granularity.value,
            message=f"Retrieved {len(trends_response)} data points"
        )
        
    except Exception as e:
        logger.error(f"Failed to get metric trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.post("/start-monitoring/{experiment_id}")
async def start_realtime_monitoring(
    experiment_id: str,
    background_tasks: BackgroundTasks
):
    """启动实时监控"""
    try:
        service = await get_realtime_metrics_service()
        
        # 启动后台更新任务
        background_tasks.add_task(
            service.start_background_updates,
            experiment_id
        )
        
        return {
            "status": "started",
            "experiment_id": experiment_id,
            "message": f"Started realtime monitoring for experiment {experiment_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/stop-monitoring")
async def stop_realtime_monitoring():
    """停止实时监控"""
    try:
        service = await get_realtime_metrics_service()
        
        await service.stop_background_updates()
        
        return {
            "status": "stopped",
            "message": "Stopped all realtime monitoring tasks"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/metrics-catalog")
async def get_metrics_catalog():
    """获取指标目录"""
    try:
        service = await get_realtime_metrics_service()
        
        # 获取所有注册的指标
        metrics = service._metrics_definitions
        
        # 按类别分组
        catalog = {
            "primary": [],
            "secondary": [],
            "guardrail": [],
            "diagnostic": []
        }
        
        for metric_name, metric_def in metrics.items():
            catalog[metric_def.category.value].append(metric_def.to_dict())
        
        return {
            "catalog": catalog,
            "total_metrics": len(metrics),
            "message": "Metrics catalog retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics catalog: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get catalog: {str(e)}")

@router.get("/metric-definition/{metric_name}")
async def get_metric_definition(metric_name: str):
    """获取指标定义"""
    try:
        service = await get_realtime_metrics_service()
        
        metric_def = service._metrics_definitions.get(metric_name)
        
        if not metric_def:
            raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
        
        return {
            "metric": metric_def.to_dict(),
            "message": f"Metric definition for '{metric_name}' retrieved"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metric definition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get definition: {str(e)}")

@router.get("/experiment/{experiment_id}/summary")
async def get_experiment_metrics_summary(
    experiment_id: str,
    time_window: TimeWindow = Query(TimeWindow.CUMULATIVE)
):
    """获取实验指标摘要"""
    try:
        service = await get_realtime_metrics_service()
        
        # 计算指标
        group_metrics = await service.calculate_metrics(
            experiment_id=experiment_id,
            time_window=time_window
        )
        
        # 生成摘要
        summary = {
            "experiment_id": experiment_id,
            "time_window": time_window.value,
            "groups": {},
            "primary_metrics": {},
            "guardrail_metrics": {}
        }
        
        for group_id, metrics in group_metrics.items():
            summary["groups"][group_id] = {
                "user_count": metrics.user_count,
                "event_count": metrics.event_count,
                "metrics_count": len(metrics.metrics)
            }
            
            # 提取关键指标
            for metric_name, snapshot in metrics.metrics.items():
                metric_def = service._metrics_definitions.get(metric_name)
                if metric_def:
                    if metric_def.category == MetricCategory.PRIMARY:
                        if metric_name not in summary["primary_metrics"]:
                            summary["primary_metrics"][metric_name] = {}
                        summary["primary_metrics"][metric_name][group_id] = snapshot.value
                    
                    elif metric_def.category == MetricCategory.GUARDRAIL:
                        if metric_name not in summary["guardrail_metrics"]:
                            summary["guardrail_metrics"][metric_name] = {}
                        summary["guardrail_metrics"][metric_name][group_id] = {
                            "value": snapshot.value,
                            "threshold_lower": metric_def.threshold_lower,
                            "threshold_upper": metric_def.threshold_upper,
                            "violated": (
                                (metric_def.threshold_lower and snapshot.value < metric_def.threshold_lower) or
                                (metric_def.threshold_upper and snapshot.value > metric_def.threshold_upper)
                            )
                        }
        
        return {
            "summary": summary,
            "message": "Experiment metrics summary generated"
        }
        
    except Exception as e:
        logger.error(f"Failed to get experiment summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.get("/health")
async def health_check():
    """实时指标服务健康检查"""
    try:
        service = await get_realtime_metrics_service()
        
        # 检查Redis连接
        redis_status = "connected" if service.redis_client else "not connected"
        
        # 检查指标定义
        metrics_count = len(service._metrics_definitions)
        
        return {
            "status": "healthy",
            "service": "realtime-metrics",
            "redis_status": redis_status,
            "registered_metrics": metrics_count,
            "message": "Realtime metrics service is running"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "realtime-metrics",
            "error": str(e),
            "message": "Realtime metrics service has issues"
        }
