from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from typing import Dict, List, Optional, Any, Union
from pydantic import Field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import asyncio
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from src.ai.model_evaluation.evaluation_engine import (
    BenchmarkManager, BenchmarkType, DifficultyLevel
)
from src.api.base_model import ApiBaseModel
from src.ai.model_evaluation.performance_monitor import PerformanceMonitor
from src.ai.model_evaluation.report_generator import (
    EvaluationReportGenerator, ReportConfig

)

from src.core.logging import get_logger
logger = get_logger(__name__)

# Pydantic模型定义
class EvaluationRequest(ApiBaseModel):
    model_name: str = Field(..., description="模型名称")
    model_path: str = Field(..., description="模型路径")
    task_type: str = Field("text_generation", description="任务类型")
    device: str = Field("auto", description="设备类型")
    batch_size: int = Field(8, description="批次大小")
    max_length: int = Field(512, description="最大长度")
    precision: str = Field("fp16", description="精度类型")
    enable_optimizations: bool = Field(True, description="启用优化")

class BenchmarkRequest(ApiBaseModel):
    name: str = Field(..., description="基准测试名称")
    tasks: List[str] = Field(..., description="任务列表")
    num_fewshot: int = Field(0, description="few-shot样本数")
    limit: Optional[int] = Field(None, description="样本限制")
    batch_size: int = Field(8, description="批次大小")
    device: str = Field("auto", description="设备")

class BatchEvaluationRequest(ApiBaseModel):
    models: List[EvaluationRequest] = Field(..., description="模型列表")
    benchmarks: List[BenchmarkRequest] = Field(..., description="基准测试列表")
    report_config: Optional[Dict[str, Any]] = Field(None, description="报告配置")

class ReportGenerationRequest(ApiBaseModel):
    evaluation_ids: List[str] = Field(..., description="评估ID列表")
    title: str = Field("模型评估报告", description="报告标题")
    subtitle: Optional[str] = Field(None, description="报告副标题")
    include_charts: bool = Field(True, description="包含图表")
    include_detailed_metrics: bool = Field(True, description="包含详细指标")
    include_recommendations: bool = Field(True, description="包含建议")
    output_format: str = Field("html", description="输出格式")

class EvaluationJob(ApiBaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_task: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    models_count: int = 0
    benchmarks_count: int = 0

# 全局状态管理
evaluation_jobs: Dict[str, EvaluationJob] = {}
benchmark_manager = BenchmarkManager()
performance_monitor = PerformanceMonitor()
report_generator = EvaluationReportGenerator()

# 启动时开始性能监控
@asynccontextmanager
async def lifespan(app):
    performance_monitor.start_monitoring()
    yield
    performance_monitor.stop_monitoring()

router = APIRouter(prefix="/model-evaluation", tags=["模型评估"])

@router.get("/", summary="模型评估系统概览")
async def get_evaluation_overview():
    """获取模型评估系统概览"""
    active_jobs = [job for job in evaluation_jobs.values() if job.status == "running"]
    completed_jobs = [job for job in evaluation_jobs.values() if job.status == "completed"]
    failed_jobs = [job for job in evaluation_jobs.values() if job.status == "failed"]
    
    system_metrics = performance_monitor.get_system_metrics_summary(60)
    active_alerts = performance_monitor.get_active_alerts()
    
    return {
        "system_status": "online",
        "jobs": {
            "active": len(active_jobs),
            "completed": len(completed_jobs),
            "failed": len(failed_jobs),
            "total": len(evaluation_jobs)
        },
        "system_metrics": system_metrics,
        "active_alerts": len(active_alerts),
        "available_benchmarks": len(benchmark_manager.benchmarks),
        "available_suites": len(benchmark_manager.suites)
    }

@router.get("/benchmarks", summary="获取可用基准测试")
async def list_benchmarks(
    benchmark_type: Optional[BenchmarkType] = Query(None, description="基准测试类型"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="难度级别"),
    language: Optional[str] = Query(None, description="语言")
):
    """获取可用的基准测试列表"""
    benchmarks = benchmark_manager.list_benchmarks(benchmark_type, difficulty, language)
    
    return {
        "benchmarks": [
            {
                "name": bench.name,
                "display_name": bench.display_name,
                "description": bench.description,
                "type": bench.benchmark_type.value,
                "difficulty": bench.difficulty.value,
                "tasks": bench.tasks,
                "languages": bench.languages,
                "num_samples": bench.num_samples,
                "estimated_runtime_minutes": bench.estimated_runtime_minutes,
                "memory_requirements_gb": bench.memory_requirements_gb,
                "metrics": bench.metrics
            }
            for bench in benchmarks
        ]
    }

@router.get("/benchmark-suites", summary="获取基准测试套件")
async def list_benchmark_suites():
    """获取基准测试套件列表"""
    suites = benchmark_manager.list_suites()
    
    return {
        "suites": [
            {
                "name": suite.name,
                "description": suite.description,
                "total_tasks": suite.total_tasks,
                "estimated_runtime_hours": suite.estimated_runtime_hours,
                "benchmarks": [
                    {
                        "name": bench.name,
                        "display_name": bench.display_name,
                        "type": bench.benchmark_type.value,
                        "difficulty": bench.difficulty.value
                    }
                    for bench in suite.benchmarks
                ]
            }
            for suite in suites
        ]
    }

@router.post("/evaluate", summary="创建评估任务")
async def create_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """创建单个模型评估任务"""
    job_id = str(uuid.uuid4())
    
    # 创建任务记录
    job = EvaluationJob(
        job_id=job_id,
        status="pending",
        created_at=utc_now(),
        models_count=1,
        benchmarks_count=1
    )
    evaluation_jobs[job_id] = job
    
    # 后台执行评估
    background_tasks.add_task(
        run_single_evaluation,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "评估任务已创建，正在后台执行"
    }

@router.post("/batch-evaluate", summary="批量评估任务")
async def create_batch_evaluation(request: BatchEvaluationRequest, background_tasks: BackgroundTasks):
    """创建批量模型评估任务"""
    job_id = str(uuid.uuid4())
    
    # 创建任务记录
    job = EvaluationJob(
        job_id=job_id,
        status="pending",
        created_at=utc_now(),
        models_count=len(request.models),
        benchmarks_count=len(request.benchmarks)
    )
    evaluation_jobs[job_id] = job
    
    # 后台执行批量评估
    background_tasks.add_task(
        run_batch_evaluation,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "models_count": len(request.models),
        "benchmarks_count": len(request.benchmarks),
        "message": "批量评估任务已创建，正在后台执行"
    }

@router.get("/jobs", summary="获取评估任务列表")
async def list_evaluation_jobs(
    status: Optional[str] = Query(None, description="任务状态筛选"),
    limit: int = Query(20, description="返回数量限制"),
    offset: int = Query(0, description="偏移量")
):
    """获取评估任务列表"""
    jobs_list = list(evaluation_jobs.values())
    
    # 状态筛选
    if status:
        jobs_list = [job for job in jobs_list if job.status == status]
    
    # 排序（最新的在前）
    jobs_list.sort(key=lambda x: x.created_at, reverse=True)
    
    # 分页
    total = len(jobs_list)
    jobs_list = jobs_list[offset:offset + limit]
    
    return {
        "jobs": [job.model_dump() for job in jobs_list],
        "total": total,
        "limit": limit,
        "offset": offset
    }

@router.get("/history", summary="获取评估历史")
async def get_evaluation_history(
    model_name: Optional[str] = Query(None, description="模型名称筛选"),
    status: Optional[str] = Query(None, description="状态筛选"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """获取评估历史记录"""
    records: List[Dict[str, Any]] = []

    for job in evaluation_jobs.values():
        if job.results:
            for result in job.results:
                metrics = result.get("metrics") or {}
                score = (
                    metrics.get("accuracy")
                    or metrics.get("f1_score")
                    or metrics.get("bleu_score")
                    or 0
                )
                records.append({
                    "id": job.job_id,
                    "model_name": result.get("model_name") or "unknown",
                    "benchmark": result.get("benchmark_name"),
                    "timestamp": result.get("timestamp") or job.created_at.isoformat(),
                    "status": job.status,
                    "score": score
                })
        else:
            records.append({
                "id": job.job_id,
                "model_name": "unknown",
                "benchmark": None,
                "timestamp": job.created_at.isoformat(),
                "status": job.status,
                "score": 0
            })

    if model_name:
        records = [r for r in records if r.get("model_name") == model_name]
    if status:
        records = [r for r in records if r.get("status") == status]

    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    total = len(records)
    paged = records[offset:offset + limit]

    return {
        "evaluations": paged,
        "total": total,
        "page": (offset // limit) + 1 if limit else 1,
        "limit": limit
    }

@router.get("/jobs/{job_id}", summary="获取评估任务详情")
async def get_evaluation_job(job_id: str):
    """获取特定评估任务的详情"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    job = evaluation_jobs[job_id]
    return job.model_dump()

@router.delete("/jobs/{job_id}", summary="取消评估任务")
async def cancel_evaluation_job(job_id: str):
    """取消评估任务"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    job = evaluation_jobs[job_id]
    
    if job.status == "completed":
        raise HTTPException(status_code=400, detail="任务已完成，无法取消")
    
    if job.status == "failed":
        raise HTTPException(status_code=400, detail="任务已失败，无需取消")
    
    job.status = "cancelled"
    job.completed_at = utc_now()
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "任务已取消"
    }

@router.post("/generate-report", summary="生成评估报告")
async def generate_evaluation_report(request: ReportGenerationRequest, background_tasks: BackgroundTasks):
    """生成评估报告"""
    if request.output_format.lower() != "html":
        raise HTTPException(status_code=400, detail="当前仅支持html格式报告")

    # 验证评估ID
    results = []
    for eval_id in request.evaluation_ids:
        if eval_id not in evaluation_jobs:
            raise HTTPException(status_code=404, detail=f"评估任务 {eval_id} 不存在")
        
        job = evaluation_jobs[eval_id]
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"评估任务 {eval_id} 尚未完成")
        
        results.extend(job.results)
    
    if not results:
        raise HTTPException(status_code=400, detail="没有可用的评估结果")
    
    # 创建报告任务
    report_job_id = str(uuid.uuid4())
    
    # 后台生成报告
    background_tasks.add_task(
        generate_report_task,
        report_job_id,
        results,
        request
    )
    
    return {
        "report_job_id": report_job_id,
        "status": "generating",
        "message": "报告生成任务已启动"
    }

@router.get("/reports/{report_id}", summary="获取生成的报告")
async def get_evaluation_report(report_id: str):
    """获取生成的评估报告"""
    report_path = Path(f"reports/{report_id}.html")
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在或尚未生成完成")
    
    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=f"evaluation_report_{report_id}.html"
    )

@router.get("/performance/system", summary="获取系统性能指标")
async def get_system_performance(time_range_minutes: int = Query(60, description="时间范围(分钟)")):
    """获取系统性能指标"""
    return performance_monitor.get_system_metrics_summary(time_range_minutes)

@router.get("/performance/models/{model_name}", summary="获取模型性能指标")
async def get_model_performance(
    model_name: str,
    time_range_minutes: int = Query(60, description="时间范围(分钟)")
):
    """获取特定模型的性能指标"""
    return performance_monitor.get_model_performance_summary(model_name, time_range_minutes)

@router.get("/performance/comparison", summary="模型性能对比")
async def compare_model_performance(
    benchmark_name: str = Query(..., description="基准测试名称"),
    model_names: List[str] = Query(..., description="模型名称列表")
):
    """对比多个模型在特定基准测试上的性能"""
    return performance_monitor.get_benchmark_comparison(benchmark_name, model_names)

@router.get("/alerts", summary="获取性能告警")
async def get_performance_alerts(
    severity: Optional[str] = Query(None, description="告警严重程度"),
    limit: int = Query(50, description="返回数量限制")
):
    """获取性能告警列表"""
    alerts = performance_monitor.get_active_alerts(severity)[:limit]
    
    return {
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "category": alert.category,
                "title": alert.title,
                "description": alert.description,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "model_name": alert.model_name,
                "benchmark_name": alert.benchmark_name,
                "resolved": alert.resolved
            }
            for alert in alerts
        ]
    }

@router.post("/alerts/{alert_id}/resolve", summary="解决告警")
async def resolve_alert(alert_id: str):
    """标记告警为已解决"""
    performance_monitor.resolve_alert(alert_id)
    
    return {
        "alert_id": alert_id,
        "status": "resolved",
        "message": "告警已标记为已解决"
    }

@router.get("/export/metrics", summary="导出性能指标")
async def export_performance_metrics(
    time_range_hours: int = Query(24, description="时间范围(小时)"),
    format_type: str = Query("json", description="导出格式")
):
    """导出性能指标数据"""
    if format_type not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="不支持的导出格式")
    
    export_path = f"exports/metrics_{utc_now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
    Path("exports").mkdir(exist_ok=True)
    
    performance_monitor.export_metrics(export_path, time_range_hours)
    
    return FileResponse(
        path=export_path,
        media_type="application/json" if format_type == "json" else "text/csv",
        filename=Path(export_path).name
    )

# 后台任务函数
async def run_single_evaluation(job_id: str, request: EvaluationRequest):
    """运行单个评估任务"""
    job = evaluation_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = utc_now()
        job.current_task = "准备评估环境"
        
        # 创建评估配置
        config = EvaluationConfig(
            model_name=request.model_name,
            model_path=request.model_path,
            task_type=request.task_type,
            device=request.device,
            batch_size=request.batch_size,
            max_length=request.max_length,
            precision=request.precision,
            enable_optimizations=request.enable_optimizations
        )
        
        # 创建评估引擎
        engine = ModelEvaluationEngine(config)
        
        job.current_task = "加载模型"
        job.progress = 0.2
        
        # 这里应该根据实际需求添加基准测试
        # 为演示目的，使用默认的基准测试配置
        benchmark_config = BenchmarkConfig(
            name="default",
            tasks=["hellaswag"],  # 示例任务
            num_fewshot=0,
            batch_size=request.batch_size,
            device=request.device
        )
        
        job.current_task = "执行评估"
        job.progress = 0.5
        
        # 执行评估
        results = engine.evaluate_with_lm_eval(benchmark_config)
        
        job.current_task = "处理结果"
        job.progress = 0.9
        
        # 保存结果
        job.results = [
            {
                "model_name": result.model_name,
                "benchmark_name": result.benchmark_name,
                "task_name": result.task_name,
                "metrics": {
                    "accuracy": result.metrics.accuracy,
                    "f1_score": result.metrics.f1_score,
                    "bleu_score": result.metrics.bleu_score,
                    "rouge_scores": result.metrics.rouge_scores,
                    "perplexity": result.metrics.perplexity,
                    "inference_time": result.metrics.inference_time,
                    "memory_usage": result.metrics.memory_usage,
                    "throughput": result.metrics.throughput
                },
                "timestamp": result.timestamp.isoformat(),
                "duration": result.duration,
                "samples_evaluated": result.samples_evaluated,
                "error": result.error
            }
            for result in results
        ]
        
        job.status = "completed"
        job.progress = 1.0
        job.completed_at = utc_now()
        job.current_task = None
        
        # 清理资源
        engine.cleanup()
        
    except Exception as e:
        logger.error(f"评估任务 {job_id} 执行失败: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = utc_now()
        job.current_task = None

async def run_batch_evaluation(job_id: str, request: BatchEvaluationRequest):
    """运行批量评估任务"""
    job = evaluation_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = utc_now()
        job.current_task = "准备批量评估"
        
        # 转换模型配置
        model_configs = [
            EvaluationConfig(
                model_name=model.model_name,
                model_path=model.model_path,
                task_type=model.task_type,
                device=model.device,
                batch_size=model.batch_size,
                max_length=model.max_length,
                precision=model.precision,
                enable_optimizations=model.enable_optimizations
            )
            for model in request.models
        ]
        
        # 转换基准测试配置
        benchmark_configs = [
            BenchmarkConfig(
                name=bench.name,
                tasks=bench.tasks,
                num_fewshot=bench.num_fewshot,
                limit=bench.limit,
                batch_size=bench.batch_size,
                device=bench.device
            )
            for bench in request.benchmarks
        ]
        
        job.current_task = "执行批量评估"
        job.progress = 0.1
        
        # 使用批量评估管理器
        batch_manager = BatchEvaluationManager(max_concurrent_evaluations=2)
        results_dict = await batch_manager.evaluate_multiple_models(
            model_configs, benchmark_configs
        )
        
        job.current_task = "处理结果"
        job.progress = 0.9
        
        # 整合所有结果
        all_results = []
        for model_name, model_results in results_dict.items():
            for result in model_results:
                all_results.append({
                    "model_name": result.model_name,
                    "benchmark_name": result.benchmark_name,
                    "task_name": result.task_name,
                    "metrics": {
                        "accuracy": result.metrics.accuracy,
                        "f1_score": result.metrics.f1_score,
                        "bleu_score": result.metrics.bleu_score,
                        "rouge_scores": result.metrics.rouge_scores,
                        "perplexity": result.metrics.perplexity,
                        "inference_time": result.metrics.inference_time,
                        "memory_usage": result.metrics.memory_usage,
                        "throughput": result.metrics.throughput
                    },
                    "timestamp": result.timestamp.isoformat(),
                    "duration": result.duration,
                    "samples_evaluated": result.samples_evaluated,
                    "error": result.error
                })
        
        job.results = all_results
        job.status = "completed"
        job.progress = 1.0
        job.completed_at = utc_now()
        job.current_task = None
        
    except Exception as e:
        logger.error(f"批量评估任务 {job_id} 执行失败: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = utc_now()
        job.current_task = None

async def generate_report_task(report_job_id: str, results: List[Dict], request: ReportGenerationRequest):
    """生成报告任务"""
    try:
        # 转换结果格式
        evaluation_results: List[EvaluationResult] = []
        for result_data in results:
            if not isinstance(result_data, dict):
                raise ValueError("评估结果格式错误")
            model_name = result_data.get("model_name")
            benchmark_name = result_data.get("benchmark_name")
            task_name = result_data.get("task_name")
            metrics_data = result_data.get("metrics")
            if not model_name or not benchmark_name or not task_name or not isinstance(metrics_data, dict):
                raise ValueError("评估结果缺少必要字段")

            metrics = EvaluationMetrics(
                accuracy=float(metrics_data.get("accuracy", 0.0)),
                f1_score=metrics_data.get("f1_score"),
                bleu_score=metrics_data.get("bleu_score"),
                rouge_scores=metrics_data.get("rouge_scores"),
                perplexity=metrics_data.get("perplexity"),
                inference_time=float(metrics_data.get("inference_time", 0.0) or 0.0),
                memory_usage=float(metrics_data.get("memory_usage", 0.0) or 0.0),
                throughput=float(metrics_data.get("throughput", 0.0) or 0.0),
            )

            ts_raw = result_data.get("timestamp")
            timestamp = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else utc_now()
            duration = float(result_data.get("duration", 0.0) or 0.0)
            samples_evaluated = int(result_data.get("samples_evaluated", 0) or 0)

            evaluation_results.append(
                EvaluationResult(
                    model_name=str(model_name),
                    benchmark_name=str(benchmark_name),
                    task_name=str(task_name),
                    metrics=metrics,
                    config=EvaluationConfig(model_name=str(model_name)),
                    timestamp=timestamp,
                    duration=duration,
                    error=result_data.get("error"),
                    samples_evaluated=samples_evaluated,
                    hardware_info=result_data.get("hardware_info"),
                )
            )
        
        # 创建报告配置
        report_config = ReportConfig(
            title=request.title,
            subtitle=request.subtitle,
            include_charts=request.include_charts,
            include_detailed_metrics=request.include_detailed_metrics,
            include_recommendations=request.include_recommendations,
            output_format=request.output_format
        )
        
        # 生成报告
        report_content = report_generator.generate_evaluation_report(
            evaluation_results,
            report_config,
            benchmark_manager,
            performance_monitor
        )
        
        # 保存报告
        report_path = Path(f"reports/{report_job_id}.html")
        report_path.parent.mkdir(exist_ok=True)
        
        report_generator.save_report(report_content, str(report_path), "html")
        
        logger.info(f"报告 {report_job_id} 生成完成")
        
    except Exception as e:
        logger.error(f"报告生成失败 {report_job_id}: {e}")
