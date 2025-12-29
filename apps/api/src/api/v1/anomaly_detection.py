"""
异常检测API端点
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import Field
from datetime import datetime
import asyncio
from email.utils import parsedate_to_datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, parse_iso_string, to_utc
from src.services.anomaly_detection_service import (
    AnomalyDetectionService,
    AnomalyType,
    DetectionMethod,
    DetectionConfig,
    Anomaly
)
from src.api.base_model import ApiBaseModel
from src.services.realtime_metrics_service import (
    AnomalyType,
    DetectionMethod,
    DetectionConfig,
    Anomaly
)

router = APIRouter(prefix="/anomalies", tags=["Anomaly Detection"])

# 服务实例
anomaly_service = AnomalyDetectionService()
_active_monitors: Dict[str, Dict[str, Any]] = {}

class DetectAnomaliesRequest(ApiBaseModel):
    """检测异常请求"""
    experiment_id: str = Field(..., description="实验ID")
    metric_name: str = Field(..., description="指标名称")
    values: List[float] = Field(..., description="指标值列表")
    timestamps: Optional[List[datetime]] = Field(None, description="时间戳列表")
    variant: Optional[str] = Field(None, description="变体名称")
    methods: Optional[List[DetectionMethod]] = Field(
        None,
        description="检测方法列表"
    )

class SRMCheckRequest(ApiBaseModel):
    """SRM检查请求"""
    experiment_id: str = Field(..., description="实验ID")
    control_count: int = Field(..., ge=0, description="对照组样本数")
    treatment_count: int = Field(..., ge=0, description="实验组样本数")
    expected_ratio: float = Field(0.5, ge=0, le=1, description="预期比例")

class DataQualityCheckRequest(ApiBaseModel):
    """数据质量检查请求"""
    experiment_id: str = Field(..., description="实验ID")
    missing_rate: float = Field(0, ge=0, le=1, description="缺失率")
    duplicate_rate: float = Field(0, ge=0, le=1, description="重复率")
    null_count: int = Field(0, ge=0, description="空值数量")
    total_count: int = Field(..., ge=0, description="总数量")

class ConfigureDetectionRequest(ApiBaseModel):
    """配置检测请求"""
    methods: List[DetectionMethod] = Field(
        [DetectionMethod.Z_SCORE, DetectionMethod.IQR],
        description="检测方法"
    )
    sensitivity: float = Field(0.95, ge=0.5, le=1.0, description="灵敏度")
    window_size: int = Field(100, ge=10, description="窗口大小")
    min_samples: int = Field(30, ge=10, description="最小样本数")
    z_threshold: float = Field(3.0, ge=2.0, description="Z-score阈值")
    iqr_multiplier: float = Field(1.5, ge=1.0, description="IQR乘数")
    enable_seasonal: bool = Field(True, description="启用季节性检测")
    enable_trend: bool = Field(True, description="启用趋势检测")

class RealTimeMonitorRequest(ApiBaseModel):
    """实时监控请求"""
    experiment_id: str = Field(..., description="实验ID")
    metrics: List[str] = Field(..., description="要监控的指标列表")
    check_interval: int = Field(60, ge=10, description="检查间隔(秒)")
    alert_threshold: str = Field("medium", description="告警阈值级别")

@router.post("/detect")
async def detect_anomalies(request: DetectAnomaliesRequest) -> Dict[str, Any]:
    """
    检测指标异常
    
    使用多种算法检测时间序列数据中的异常
    """
    try:
        # 如果指定了检测方法，更新配置
        if request.methods:
            config = DetectionConfig(methods=request.methods)
            anomaly_service.config = config
            
        anomalies = await anomaly_service.detect_anomalies(
            experiment_id=request.experiment_id,
            metric_name=request.metric_name,
            values=request.values,
            timestamps=request.timestamps,
            variant=request.variant
        )
        
        return {
            "success": True,
            "anomalies": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.type,
                    "severity": a.severity,
                    "metric": a.metric_name,
                    "variant": a.variant,
                    "observed": a.observed_value,
                    "expected": a.expected_value,
                    "deviation": a.deviation,
                    "method": a.detection_method,
                    "confidence": a.confidence,
                    "description": a.description,
                    "metadata": a.metadata
                }
                for a in anomalies
            ],
            "total_count": len(anomalies),
            "methods_used": list(set(a.detection_method for a in anomalies))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-srm")
async def check_sample_ratio_mismatch(request: SRMCheckRequest) -> Dict[str, Any]:
    """
    检查样本比例不匹配(SRM)
    
    验证实验分组是否符合预期比例
    """
    try:
        anomaly = await anomaly_service.detect_sample_ratio_mismatch(
            experiment_id=request.experiment_id,
            control_count=request.control_count,
            treatment_count=request.treatment_count,
            expected_ratio=request.expected_ratio
        )
        
        if anomaly:
            return {
                "success": True,
                "has_srm": True,
                "anomaly": {
                    "severity": anomaly.severity,
                    "observed_ratio": anomaly.observed_value,
                    "expected_ratio": anomaly.expected_value,
                    "confidence": anomaly.confidence,
                    "description": anomaly.description,
                    "metadata": anomaly.metadata
                }
            }
        else:
            return {
                "success": True,
                "has_srm": False,
                "message": "样本比例正常"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-data-quality")
async def check_data_quality(request: DataQualityCheckRequest) -> Dict[str, Any]:
    """
    检查数据质量
    
    检测缺失值、重复值等数据质量问题
    """
    try:
        data = {
            "missing_rate": request.missing_rate,
            "duplicate_rate": request.duplicate_rate,
            "null_count": request.null_count,
            "total_count": request.total_count
        }
        
        # 计算额外的质量指标
        if request.total_count > 0:
            data["null_rate"] = request.null_count / request.total_count
        else:
            data["null_rate"] = 0
            
        anomalies = await anomaly_service.detect_data_quality_issues(
            request.experiment_id,
            data
        )
        
        return {
            "success": True,
            "quality_issues": [
                {
                    "type": a.type,
                    "severity": a.severity,
                    "description": a.description,
                    "value": a.observed_value,
                    "metadata": a.metadata
                }
                for a in anomalies
            ],
            "has_issues": len(anomalies) > 0,
            "quality_score": max(0, 1 - sum(a.observed_value for a in anomalies))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/{experiment_id}")
async def get_anomaly_summary(
    experiment_id: str,
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间")
) -> Dict[str, Any]:
    """
    获取异常摘要
    
    返回指定时间范围内的异常统计
    """
    try:
        parsed_start_time = _parse_datetime_param(start_time, "开始时间")
        parsed_end_time = _parse_datetime_param(end_time, "结束时间")
        summary = await anomaly_service.get_anomaly_summary(
            experiment_id=experiment_id,
            start_time=parsed_start_time,
            end_time=parsed_end_time
        )
        
        return {
            "success": True,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _parse_datetime_param(value: Optional[str], field_name: str) -> Optional[datetime]:
    if value is None or value == "":
        return None
    parsed = parse_iso_string(value)
    if parsed:
        return parsed
    try:
        parsed_http = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed_http = None
    if parsed_http is None:
        raise HTTPException(status_code=422, detail=f"{field_name}格式无效")
    return to_utc(parsed_http)

@router.post("/configure")
async def configure_detection(request: ConfigureDetectionRequest) -> Dict[str, Any]:
    """
    配置检测参数
    
    自定义异常检测的灵敏度和方法
    """
    try:
        config = DetectionConfig(
            methods=request.methods,
            sensitivity=request.sensitivity,
            window_size=request.window_size,
            min_samples=request.min_samples,
            z_threshold=request.z_threshold,
            iqr_multiplier=request.iqr_multiplier,
            enable_seasonal=request.enable_seasonal,
            enable_trend=request.enable_trend
        )
        
        anomaly_service.config = config
        
        return {
            "success": True,
            "message": "检测配置已更新",
            "config": {
                "methods": [m.value for m in config.methods],
                "sensitivity": config.sensitivity,
                "window_size": config.window_size,
                "min_samples": config.min_samples,
                "z_threshold": config.z_threshold,
                "iqr_multiplier": config.iqr_multiplier,
                "enable_seasonal": config.enable_seasonal,
                "enable_trend": config.enable_trend
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/real-time-monitor")
async def setup_real_time_monitoring(
    request: RealTimeMonitorRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    设置实时监控
    
    配置指标的实时异常监控
    """
    try:
        realtime_service = await get_realtime_metrics_service()
        monitor_key = request.experiment_id

        async def monitor_task():
            while True:
                try:
                    for metric in request.metrics:
                        trends = await realtime_service.get_metric_trends(
                            request.experiment_id,
                            metric,
                            granularity=TimeWindow.REALTIME,
                        )
                        if not trends:
                            continue
                        values = [point.value for point in trends]
                        timestamps = [point.timestamp for point in trends]
                        await anomaly_service.detect_anomalies(
                            experiment_id=request.experiment_id,
                            metric_name=metric,
                            values=values,
                            timestamps=timestamps,
                        )
                    await asyncio.sleep(request.check_interval)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(request.check_interval)

        existing = _active_monitors.get(monitor_key)
        if existing:
            if (
                existing.get("metrics") != request.metrics
                or existing.get("check_interval") != request.check_interval
                or existing.get("alert_threshold") != request.alert_threshold
            ):
                existing["task"].cancel()
                _active_monitors.pop(monitor_key, None)
            else:
                return {
                    "success": True,
                    "message": "实时监控已在运行",
                    "experiment_id": request.experiment_id,
                    "metrics": request.metrics,
                    "check_interval": request.check_interval,
                    "alert_threshold": request.alert_threshold
                }

        task = asyncio.create_task(monitor_task())
        _active_monitors[monitor_key] = {
            "task": task,
            "metrics": request.metrics,
            "check_interval": request.check_interval,
            "alert_threshold": request.alert_threshold,
        }
        background_tasks.add_task(realtime_service.start_background_updates, request.experiment_id)

        return {
            "success": True,
            "message": "实时监控已启动",
            "experiment_id": request.experiment_id,
            "metrics": request.metrics,
            "check_interval": request.check_interval,
            "alert_threshold": request.alert_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def list_anomaly_types() -> Dict[str, Any]:
    """列出所有异常类型"""
    return {
        "success": True,
        "types": [
            {
                "value": t.value,
                "name": t.value.replace("_", " ").title(),
                "description": _get_type_description(t)
            }
            for t in AnomalyType
        ]
    }

@router.get("/methods")
async def list_detection_methods() -> Dict[str, Any]:
    """列出所有检测方法"""
    return {
        "success": True,
        "methods": [
            {
                "value": m.value,
                "name": m.value.replace("_", " ").title(),
                "description": _get_method_description(m)
            }
            for m in DetectionMethod
        ]
    }

@router.post("/batch-detect")
async def batch_detect_anomalies(
    experiments: List[str] = Query(..., description="实验ID列表"),
    metrics: List[str] = Query(..., description="指标列表")
) -> Dict[str, Any]:
    """
    批量检测异常
    
    对多个实验和指标进行异常检测
    """
    try:
        results = {}
        realtime_service = await get_realtime_metrics_service()
        
        for exp_id in experiments:
            exp_results = {}
            for metric in metrics:
                try:
                    trends = await realtime_service.get_metric_trends(
                        exp_id,
                        metric,
                        granularity=TimeWindow.HOURLY
                    )
                except Exception:
                    trends = []

                values = [t.value for t in trends]
                timestamps = [t.timestamp for t in trends]
                
                anomalies = await anomaly_service.detect_anomalies(
                    experiment_id=exp_id,
                    metric_name=metric,
                    values=values,
                    timestamps=timestamps if timestamps else None
                )
                
                exp_results[metric] = {
                    "anomaly_count": len(anomalies),
                    "severities": [a.severity for a in anomalies],
                    "sample_size": len(values)
                }
                
            results[exp_id] = exp_results
            
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_type_description(anomaly_type: AnomalyType) -> str:
    """获取异常类型描述"""
    descriptions = {
        AnomalyType.METRIC_SPIKE: "指标值突然大幅上升",
        AnomalyType.METRIC_DROP: "指标值突然大幅下降",
        AnomalyType.SAMPLE_RATIO_MISMATCH: "实验分组比例与预期不符",
        AnomalyType.DATA_QUALITY: "数据质量问题，如缺失值或重复值",
        AnomalyType.SEASONALITY: "季节性模式异常",
        AnomalyType.TREND_CHANGE: "趋势发生显著变化",
        AnomalyType.OUTLIER: "离群值或极端值",
        AnomalyType.VARIANCE_CHANGE: "方差发生显著变化",
        AnomalyType.DISTRIBUTION_SHIFT: "数据分布发生偏移",
        AnomalyType.CORRELATION_BREAK: "相关性模式被破坏"
    }
    return descriptions.get(anomaly_type, "未知异常类型")

def _get_method_description(method: DetectionMethod) -> str:
    """获取检测方法描述"""
    descriptions = {
        DetectionMethod.Z_SCORE: "基于标准差的统计方法",
        DetectionMethod.IQR: "基于四分位距的稳健方法",
        DetectionMethod.ISOLATION_FOREST: "基于隔离的机器学习方法",
        DetectionMethod.LOCAL_OUTLIER_FACTOR: "基于局部密度的方法",
        DetectionMethod.DBSCAN: "基于密度的聚类方法",
        DetectionMethod.STATISTICAL_PROCESS_CONTROL: "统计过程控制图",
        DetectionMethod.EXPONENTIAL_SMOOTHING: "指数平滑预测方法",
        DetectionMethod.PROPHET: "Facebook时间序列预测模型",
        DetectionMethod.CUSUM: "累积和控制图",
        DetectionMethod.EWMA: "指数加权移动平均"
    }
    return descriptions.get(method, "未知检测方法")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "anomaly_detection",
        "status": "healthy",
        "config": {
            "methods": [m.value for m in anomaly_service.config.methods],
            "sensitivity": anomaly_service.config.sensitivity
        }
    }
