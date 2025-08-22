"""
实验指标实时计算服务 - 实时监控和分析A/B测试指标
"""
import asyncio
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import redis.asyncio as redis

from core.logging import get_logger
from core.database import get_db
from services.statistical_analysis_service import (
    get_stats_calculator,
    MetricType,
    DescriptiveStats
)
from services.hypothesis_testing_service import (
    get_hypothesis_testing_service,
    HypothesisType
)
from services.confidence_interval_service import (
    get_confidence_interval_service,
    ParameterType
)

logger = get_logger(__name__)


class MetricCategory(str, Enum):
    """指标类别"""
    PRIMARY = "primary"  # 主要指标
    SECONDARY = "secondary"  # 次要指标
    GUARDRAIL = "guardrail"  # 护栏指标
    DIAGNOSTIC = "diagnostic"  # 诊断指标


class AggregationType(str, Enum):
    """聚合类型"""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    RATE = "rate"
    PERCENTILE = "percentile"
    UNIQUE = "unique"


class TimeWindow(str, Enum):
    """时间窗口"""
    REALTIME = "realtime"  # 实时（最近1分钟）
    HOURLY = "hourly"  # 每小时
    DAILY = "daily"  # 每天
    WEEKLY = "weekly"  # 每周
    CUMULATIVE = "cumulative"  # 累计


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str  # 指标名称
    display_name: str  # 显示名称
    metric_type: MetricType  # 指标类型
    category: MetricCategory  # 指标类别
    aggregation: AggregationType  # 聚合类型
    unit: str = ""  # 单位
    description: str = ""  # 描述
    formula: Optional[str] = None  # 计算公式
    numerator_event: Optional[str] = None  # 分子事件
    denominator_event: Optional[str] = None  # 分母事件
    threshold_lower: Optional[float] = None  # 下限阈值
    threshold_upper: Optional[float] = None  # 上限阈值
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "metric_type": self.metric_type.value,
            "category": self.category.value,
            "aggregation": self.aggregation.value,
            "unit": self.unit,
            "description": self.description,
            "formula": self.formula,
            "numerator_event": self.numerator_event,
            "denominator_event": self.denominator_event,
            "threshold_lower": self.threshold_lower,
            "threshold_upper": self.threshold_upper
        }


@dataclass
class MetricSnapshot:
    """指标快照"""
    metric_name: str  # 指标名称
    timestamp: datetime  # 时间戳
    value: float  # 值
    sample_size: int  # 样本量
    confidence_interval: Optional[Tuple[float, float]] = None  # 置信区间
    variance: Optional[float] = None  # 方差
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metric_name": self.metric_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "sample_size": self.sample_size,
            "confidence_interval": self.confidence_interval,
            "variance": self.variance
        }


@dataclass
class GroupMetrics:
    """分组指标"""
    group_id: str  # 分组ID
    group_name: str  # 分组名称
    metrics: Dict[str, MetricSnapshot]  # 指标快照字典
    user_count: int  # 用户数
    event_count: int  # 事件数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "user_count": self.user_count,
            "event_count": self.event_count
        }


@dataclass
class MetricComparison:
    """指标比较结果"""
    metric_name: str  # 指标名称
    control_value: float  # 对照组值
    treatment_value: float  # 实验组值
    absolute_difference: float  # 绝对差异
    relative_difference: float  # 相对差异（%）
    p_value: Optional[float] = None  # p值
    is_significant: bool = False  # 是否显著
    confidence_interval: Optional[Tuple[float, float]] = None  # 差异置信区间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metric_name": self.metric_name,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval
        }


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.logger = logger
        self.stats_calculator = get_stats_calculator()
        self.hypothesis_service = get_hypothesis_testing_service()
        self.confidence_service = get_confidence_interval_service()
    
    async def calculate_conversion_rate(self, conversions: int, total_users: int) -> MetricSnapshot:
        """计算转化率"""
        try:
            if total_users == 0:
                return MetricSnapshot(
                    metric_name="conversion_rate",
                    timestamp=datetime.now(timezone.utc),
                    value=0.0,
                    sample_size=0
                )
            
            rate = conversions / total_users
            
            # 计算置信区间
            ci_result = self.confidence_service.calculate_confidence_interval(
                parameter_type=ParameterType.PROPORTION,
                successes=conversions,
                total=total_users,
                confidence_level=0.95
            )
            
            return MetricSnapshot(
                metric_name="conversion_rate",
                timestamp=datetime.now(timezone.utc),
                value=rate,
                sample_size=total_users,
                confidence_interval=(ci_result.lower_bound, ci_result.upper_bound),
                variance=ci_result.standard_error ** 2 if ci_result.standard_error else None
            )
            
        except Exception as e:
            self.logger.error(f"Conversion rate calculation failed: {e}")
            raise
    
    async def calculate_average_metric(self, values: List[float], metric_name: str) -> MetricSnapshot:
        """计算平均值指标"""
        try:
            if not values:
                return MetricSnapshot(
                    metric_name=metric_name,
                    timestamp=datetime.now(timezone.utc),
                    value=0.0,
                    sample_size=0
                )
            
            stats = self.stats_calculator.basic_calculator.calculate_descriptive_stats(values)
            
            # 计算置信区间
            ci_result = self.confidence_service.calculate_confidence_interval(
                parameter_type=ParameterType.MEAN,
                sample=values,
                confidence_level=0.95
            )
            
            return MetricSnapshot(
                metric_name=metric_name,
                timestamp=datetime.now(timezone.utc),
                value=stats.mean,
                sample_size=stats.count,
                confidence_interval=(ci_result.lower_bound, ci_result.upper_bound),
                variance=stats.variance
            )
            
        except Exception as e:
            self.logger.error(f"Average metric calculation failed: {e}")
            raise
    
    async def calculate_percentile_metric(self, values: List[float], percentile: float,
                                        metric_name: str) -> MetricSnapshot:
        """计算分位数指标"""
        try:
            if not values:
                return MetricSnapshot(
                    metric_name=metric_name,
                    timestamp=datetime.now(timezone.utc),
                    value=0.0,
                    sample_size=0
                )
            
            percentile_value = self.stats_calculator.basic_calculator.calculate_percentiles(
                values, [percentile]
            )[0]
            
            return MetricSnapshot(
                metric_name=metric_name,
                timestamp=datetime.now(timezone.utc),
                value=percentile_value,
                sample_size=len(values)
            )
            
        except Exception as e:
            self.logger.error(f"Percentile metric calculation failed: {e}")
            raise
    
    async def compare_metrics(self, control_snapshot: MetricSnapshot,
                            treatment_snapshot: MetricSnapshot,
                            metric_type: MetricType) -> MetricComparison:
        """比较两组指标"""
        try:
            absolute_diff = treatment_snapshot.value - control_snapshot.value
            
            # 计算相对差异
            if control_snapshot.value != 0:
                relative_diff = (absolute_diff / control_snapshot.value) * 100
            else:
                relative_diff = 0.0 if absolute_diff == 0 else float('inf')
            
            # 进行假设检验
            p_value = None
            is_significant = False
            confidence_interval = None
            
            if control_snapshot.sample_size > 0 and treatment_snapshot.sample_size > 0:
                if metric_type == MetricType.CONVERSION:
                    # 比例检验
                    test_result = self.hypothesis_service.chi_square_calculator.proportion_test(
                        successes1=int(control_snapshot.value * control_snapshot.sample_size),
                        total1=control_snapshot.sample_size,
                        successes2=int(treatment_snapshot.value * treatment_snapshot.sample_size),
                        total2=treatment_snapshot.sample_size,
                        hypothesis_type=HypothesisType.TWO_SIDED,
                        alpha=0.05
                    )
                    p_value = test_result.p_value
                    is_significant = test_result.is_significant
                
                # 计算差异的置信区间
                if metric_type == MetricType.CONVERSION:
                    ci_result = self.confidence_service.proportion_calculator.two_proportion_difference_ci(
                        successes1=int(control_snapshot.value * control_snapshot.sample_size),
                        total1=control_snapshot.sample_size,
                        successes2=int(treatment_snapshot.value * treatment_snapshot.sample_size),
                        total2=treatment_snapshot.sample_size,
                        confidence_level=0.95
                    )
                    confidence_interval = (ci_result.lower_bound, ci_result.upper_bound)
            
            return MetricComparison(
                metric_name=control_snapshot.metric_name,
                control_value=control_snapshot.value,
                treatment_value=treatment_snapshot.value,
                absolute_difference=absolute_diff,
                relative_difference=relative_diff,
                p_value=p_value,
                is_significant=is_significant,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            self.logger.error(f"Metric comparison failed: {e}")
            raise


class RealtimeMetricsService:
    """实时指标服务"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.calculator = MetricsCalculator()
        self.logger = logger
        self._metrics_definitions: Dict[str, MetricDefinition] = {}
        self._metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._update_interval = 60  # 更新间隔（秒）
        self._background_tasks = []
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化Redis连接
            if not self.redis_client:
                self.redis_client = await redis.from_url(
                    "redis://localhost:6379",
                    encoding="utf-8",
                    decode_responses=True
                )
            
            # 加载默认指标定义
            self._load_default_metrics()
            
            self.logger.info("Realtime metrics service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize realtime metrics service: {e}")
            raise
    
    def _load_default_metrics(self):
        """加载默认指标定义"""
        default_metrics = [
            MetricDefinition(
                name="conversion_rate",
                display_name="转化率",
                metric_type=MetricType.CONVERSION,
                category=MetricCategory.PRIMARY,
                aggregation=AggregationType.RATE,
                unit="%",
                description="用户转化率",
                numerator_event="conversion",
                denominator_event="exposure"
            ),
            MetricDefinition(
                name="revenue_per_user",
                display_name="用户平均收入",
                metric_type=MetricType.CONTINUOUS,
                category=MetricCategory.PRIMARY,
                aggregation=AggregationType.AVERAGE,
                unit="¥",
                description="每用户平均收入"
            ),
            MetricDefinition(
                name="click_through_rate",
                display_name="点击率",
                metric_type=MetricType.CONVERSION,
                category=MetricCategory.SECONDARY,
                aggregation=AggregationType.RATE,
                unit="%",
                description="点击率",
                numerator_event="click",
                denominator_event="impression"
            ),
            MetricDefinition(
                name="bounce_rate",
                display_name="跳出率",
                metric_type=MetricType.CONVERSION,
                category=MetricCategory.GUARDRAIL,
                aggregation=AggregationType.RATE,
                unit="%",
                description="跳出率",
                threshold_upper=0.5
            ),
            MetricDefinition(
                name="page_load_time",
                display_name="页面加载时间",
                metric_type=MetricType.CONTINUOUS,
                category=MetricCategory.GUARDRAIL,
                aggregation=AggregationType.PERCENTILE,
                unit="ms",
                description="页面加载时间P95",
                threshold_upper=3000
            ),
            MetricDefinition(
                name="sample_ratio",
                display_name="样本比例",
                metric_type=MetricType.CONTINUOUS,
                category=MetricCategory.DIAGNOSTIC,
                aggregation=AggregationType.COUNT,
                description="实验组样本比例"
            )
        ]
        
        for metric in default_metrics:
            self._metrics_definitions[metric.name] = metric
    
    def register_metric(self, metric: MetricDefinition):
        """注册指标定义"""
        self._metrics_definitions[metric.name] = metric
        self.logger.info(f"Registered metric: {metric.name}")
    
    async def calculate_metrics(self, experiment_id: str, 
                              time_window: TimeWindow = TimeWindow.CUMULATIVE) -> Dict[str, GroupMetrics]:
        """计算实验指标"""
        try:
            # 从数据库获取事件数据
            events_data = await self._fetch_events_data(experiment_id, time_window)
            
            # 按分组计算指标
            group_metrics = {}
            
            for group_id, group_events in events_data.items():
                metrics = {}
                
                # 计算每个注册的指标
                for metric_name, metric_def in self._metrics_definitions.items():
                    snapshot = await self._calculate_single_metric(
                        metric_def, group_events
                    )
                    if snapshot:
                        metrics[metric_name] = snapshot
                
                group_metrics[group_id] = GroupMetrics(
                    group_id=group_id,
                    group_name=group_events.get("name", group_id),
                    metrics=metrics,
                    user_count=len(set(e.get("user_id") for e in group_events.get("events", []))),
                    event_count=len(group_events.get("events", []))
                )
            
            # 缓存结果
            await self._cache_metrics(experiment_id, group_metrics)
            
            return group_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics for experiment {experiment_id}: {e}")
            raise
    
    async def _calculate_single_metric(self, metric_def: MetricDefinition,
                                     group_events: Dict[str, Any]) -> Optional[MetricSnapshot]:
        """计算单个指标"""
        try:
            events = group_events.get("events", [])
            
            if metric_def.metric_type == MetricType.CONVERSION:
                # 转化率指标
                if metric_def.numerator_event and metric_def.denominator_event:
                    numerator_count = sum(
                        1 for e in events 
                        if e.get("event_type") == metric_def.numerator_event
                    )
                    denominator_count = sum(
                        1 for e in events 
                        if e.get("event_type") == metric_def.denominator_event
                    )
                    
                    if denominator_count > 0:
                        return await self.calculator.calculate_conversion_rate(
                            numerator_count, denominator_count
                        )
            
            elif metric_def.metric_type == MetricType.CONTINUOUS:
                # 连续值指标
                values = [
                    float(e.get("value", 0)) 
                    for e in events 
                    if e.get("metric_name") == metric_def.name
                ]
                
                if values:
                    if metric_def.aggregation == AggregationType.AVERAGE:
                        return await self.calculator.calculate_average_metric(
                            values, metric_def.name
                        )
                    elif metric_def.aggregation == AggregationType.PERCENTILE:
                        return await self.calculator.calculate_percentile_metric(
                            values, 95.0, metric_def.name
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to calculate metric {metric_def.name}: {e}")
            return None
    
    async def compare_groups(self, experiment_id: str, control_group: str,
                           treatment_group: str) -> Dict[str, MetricComparison]:
        """比较实验组"""
        try:
            # 获取分组指标
            group_metrics = await self.calculate_metrics(experiment_id)
            
            if control_group not in group_metrics or treatment_group not in group_metrics:
                raise ValueError("Invalid group IDs")
            
            control_metrics = group_metrics[control_group].metrics
            treatment_metrics = group_metrics[treatment_group].metrics
            
            # 比较每个指标
            comparisons = {}
            
            for metric_name in control_metrics:
                if metric_name in treatment_metrics:
                    metric_def = self._metrics_definitions.get(metric_name)
                    if metric_def:
                        comparison = await self.calculator.compare_metrics(
                            control_metrics[metric_name],
                            treatment_metrics[metric_name],
                            metric_def.metric_type
                        )
                        comparisons[metric_name] = comparison
            
            return comparisons
            
        except Exception as e:
            self.logger.error(f"Failed to compare groups: {e}")
            raise
    
    async def get_metric_trends(self, experiment_id: str, metric_name: str,
                              granularity: TimeWindow = TimeWindow.HOURLY) -> List[MetricSnapshot]:
        """获取指标趋势"""
        try:
            cache_key = f"metrics:trends:{experiment_id}:{metric_name}:{granularity.value}"
            
            # 从缓存获取
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return [
                        MetricSnapshot(**item) 
                        for item in json.loads(cached_data)
                    ]
            
            # 计算趋势数据
            trends = await self._calculate_metric_trends(
                experiment_id, metric_name, granularity
            )
            
            # 缓存结果
            if self.redis_client and trends:
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5分钟缓存
                    json.dumps([t.to_dict() for t in trends])
                )
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to get metric trends: {e}")
            raise
    
    async def _fetch_events_data(self, experiment_id: str,
                               time_window: TimeWindow) -> Dict[str, Any]:
        """获取事件数据"""
        # 这里应该从数据库获取实际数据
        # 暂时返回模拟数据
        return {
            "control": {
                "name": "Control Group",
                "events": [
                    {"event_type": "exposure", "user_id": f"user_{i}"}
                    for i in range(100)
                ] + [
                    {"event_type": "conversion", "user_id": f"user_{i}"}
                    for i in range(15)
                ]
            },
            "treatment": {
                "name": "Treatment Group",
                "events": [
                    {"event_type": "exposure", "user_id": f"user_{i}"}
                    for i in range(100, 200)
                ] + [
                    {"event_type": "conversion", "user_id": f"user_{i}"}
                    for i in range(100, 119)
                ]
            }
        }
    
    async def _cache_metrics(self, experiment_id: str, metrics: Dict[str, GroupMetrics]):
        """缓存指标"""
        if self.redis_client:
            try:
                cache_key = f"metrics:realtime:{experiment_id}"
                cache_data = {
                    group_id: group_metric.to_dict()
                    for group_id, group_metric in metrics.items()
                }
                
                await self.redis_client.setex(
                    cache_key,
                    self._update_interval,
                    json.dumps(cache_data)
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to cache metrics: {e}")
    
    async def _calculate_metric_trends(self, experiment_id: str, metric_name: str,
                                     granularity: TimeWindow) -> List[MetricSnapshot]:
        """计算指标趋势"""
        # 这里应该根据粒度从数据库聚合数据
        # 暂时返回模拟数据
        trends = []
        now = datetime.now(timezone.utc)
        
        for i in range(24):
            timestamp = now - timedelta(hours=23 - i)
            value = 0.15 + (i * 0.002) + (0.01 * (i % 3))  # 模拟上升趋势
            
            trends.append(MetricSnapshot(
                metric_name=metric_name,
                timestamp=timestamp,
                value=value,
                sample_size=100 + i * 10
            ))
        
        return trends
    
    async def start_background_updates(self, experiment_id: str):
        """启动后台更新任务"""
        async def update_task():
            while True:
                try:
                    await self.calculate_metrics(experiment_id)
                    await asyncio.sleep(self._update_interval)
                except Exception as e:
                    self.logger.error(f"Background update failed: {e}")
                    await asyncio.sleep(self._update_interval)
        
        task = asyncio.create_task(update_task())
        self._background_tasks.append(task)
        
        self.logger.info(f"Started background updates for experiment {experiment_id}")
    
    async def stop_background_updates(self):
        """停止后台更新任务"""
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        
        self.logger.info("Stopped all background update tasks")
    
    async def cleanup(self):
        """清理资源"""
        await self.stop_background_updates()
        
        if self.redis_client:
            await self.redis_client.close()


# 全局实例
_realtime_metrics_service = None

async def get_realtime_metrics_service() -> RealtimeMetricsService:
    """获取实时指标服务实例（单例模式）"""
    global _realtime_metrics_service
    if _realtime_metrics_service is None:
        _realtime_metrics_service = RealtimeMetricsService()
        await _realtime_metrics_service.initialize()
    return _realtime_metrics_service