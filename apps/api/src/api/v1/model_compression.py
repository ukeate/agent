"""
模型压缩API接口

实现压缩任务提交和管理接口
添加压缩进度监控和结果查询
实现压缩模型下载和部署
提供压缩配置和模板管理
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import Field
import asyncio
import os
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.ai.model_compression import (
    CompressionPipeline,
    CompressionJob,
    CompressionResult,
    CompressionMethod,
    QuantizationConfig,
    CompressionJob,
    CompressionResult,
    CompressionMethod,
    QuantizationConfig,
    DistillationConfig,
    PruningConfig,
    CompressionStrategy,
    PipelineStatus,
    get_compression_pipeline,
    get_compression_evaluator,
    get_supported_methods,
    get_default_config,
    create_compression_job,
    DEFAULT_COMPRESSION_STRATEGIES
)

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/model-compression", tags=["模型压缩"])

# 全局压缩流水线实例
pipeline = get_compression_pipeline()
evaluator = get_compression_evaluator()

# Pydantic模型定义
class CompressionJobRequest(ApiBaseModel):
    """压缩任务请求"""
    job_name: str = Field(..., description="任务名称")
    model_path: str = Field(..., description="模型路径")
    compression_method: CompressionMethod = Field(..., description="压缩方法")
    
    # 量化配置
    quantization_method: Optional[str] = Field(None, description="量化方法")
    precision: Optional[str] = Field(None, description="量化精度")
    calibration_dataset_size: Optional[int] = Field(512, description="校准数据集大小")
    
    # 蒸馏配置
    teacher_model: Optional[str] = Field(None, description="教师模型路径")
    student_model: Optional[str] = Field(None, description="学生模型路径")
    distillation_type: Optional[str] = Field("response_based", description="蒸馏类型")
    temperature: Optional[float] = Field(3.0, description="蒸馏温度")
    alpha: Optional[float] = Field(0.5, description="蒸馏损失权重")
    
    # 剪枝配置
    pruning_type: Optional[str] = Field(None, description="剪枝类型")
    sparsity_ratio: Optional[float] = Field(0.5, description="稀疏度比例")
    importance_metric: Optional[str] = Field("magnitude", description="重要性度量")
    gradual_pruning: Optional[bool] = Field(True, description="渐进式剪枝")
    recovery_epochs: Optional[int] = Field(5, description="恢复训练轮次")
    
    # 通用配置
    output_dir: Optional[str] = Field("compressed_models", description="输出目录")
    save_intermediate: Optional[bool] = Field(False, description="保存中间结果")

class CompressionJobResponse(ApiBaseModel):
    """压缩任务响应"""
    job_id: str
    job_name: str
    status: str
    created_at: str
    message: str

class JobStatusResponse(ApiBaseModel):
    """任务状态响应"""
    job_id: str
    current_stage: str
    progress_percent: float
    estimated_time_remaining: float
    last_update: str
    recent_logs: List[str]

class CompressionResultResponse(ApiBaseModel):
    """压缩结果响应"""
    job_id: str
    compression_ratio: float
    speedup_ratio: Optional[float]
    memory_reduction: Optional[float]
    accuracy_retention: Optional[float]
    compressed_model_path: str
    evaluation_report_path: str
    compression_time: float

class BenchmarkRequest(ApiBaseModel):
    """基准测试请求"""
    model_path: str = Field(..., description="模型路径")
    device_name: Optional[str] = Field(None, description="设备名称")
    sequence_lengths: Optional[List[int]] = Field([128, 256, 512], description="序列长度列表")
    batch_sizes: Optional[List[int]] = Field([1, 2, 4], description="批次大小列表")

class StrategyRecommendationRequest(ApiBaseModel):
    """策略推荐请求"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field(..., description="模型类型")
    num_parameters: int = Field(..., description="参数数量")
    target_scenario: str = Field("cloud", description="目标场景")
    accuracy_tolerance: float = Field(0.05, description="精度容忍度")
    size_reduction_target: float = Field(0.5, description="大小减少目标")

# API路由定义

@router.post("/jobs", response_model=CompressionJobResponse)
async def create_compression_job_api(request: CompressionJobRequest) -> CompressionJobResponse:
    """创建模型压缩任务"""
    
    try:
        # 验证模型路径
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {request.model_path}")
        
        # 构建配置对象
        quantization_config = None
        if request.compression_method in [CompressionMethod.QUANTIZATION, CompressionMethod.MIXED]:
            from src.ai.model_compression import QuantizationMethod, PrecisionType
            quantization_config = QuantizationConfig(
                method=QuantizationMethod(request.quantization_method or "post_training_quantization"),
                precision=PrecisionType(request.precision or "int8"),
                calibration_dataset_size=request.calibration_dataset_size or 512
            )
        
        distillation_config = None
        if request.compression_method in [CompressionMethod.DISTILLATION, CompressionMethod.MIXED]:
            if not request.teacher_model or not request.student_model:
                raise HTTPException(status_code=400, detail="蒸馏需要指定teacher_model和student_model")
            
            distillation_config = DistillationConfig(
                teacher_model=request.teacher_model,
                student_model=request.student_model,
                distillation_type=request.distillation_type or "response_based",
                temperature=request.temperature or 3.0,
                alpha=request.alpha or 0.5
            )
        
        pruning_config = None
        if request.compression_method in [CompressionMethod.PRUNING, CompressionMethod.MIXED]:
            from src.ai.model_compression import PruningType
            pruning_config = PruningConfig(
                pruning_type=PruningType(request.pruning_type or "unstructured"),
                sparsity_ratio=request.sparsity_ratio or 0.5,
                importance_metric=request.importance_metric or "magnitude",
                gradual_pruning=request.gradual_pruning or True,
                recovery_epochs=request.recovery_epochs or 5
            )
        
        # 创建压缩任务
        job = CompressionJob(
            job_name=request.job_name,
            model_path=request.model_path,
            compression_method=request.compression_method,
            quantization_config=quantization_config,
            distillation_config=distillation_config,
            pruning_config=pruning_config,
            output_dir=request.output_dir or "compressed_models",
            save_intermediate=request.save_intermediate or False
        )
        
        # 提交任务到流水线
        job_id = await pipeline.submit_job(job)
        
        logger.info(f"创建压缩任务成功: {job_id}")
        
        return CompressionJobResponse(
            job_id=job_id,
            job_name=job.job_name,
            status=job.status,
            created_at=job.created_at.isoformat(),
            message="任务创建成功，已加入执行队列"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建压缩任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@router.get("/jobs", response_model=List[CompressionJobResponse])
async def list_compression_jobs() -> List[CompressionJobResponse]:
    """获取所有压缩任务列表"""
    
    try:
        active_jobs = pipeline.get_active_jobs()
        job_responses = []
        
        for job_id in active_jobs:
            status = await pipeline.get_job_status(job_id)
            if status:
                job_responses.append(CompressionJobResponse(
                    job_id=job_id,
                    job_name=job_id,  # 简化处理
                    status=status.current_stage,
                    created_at=status.last_update,
                    message=status.logs[-1] if status.logs else "运行中"
                ))
        
        return job_responses
    
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """获取指定任务的详细状态"""
    
    try:
        status = await pipeline.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        
        return JobStatusResponse(
            job_id=status.job_id,
            current_stage=status.current_stage,
            progress_percent=status.progress_percent,
            estimated_time_remaining=status.estimated_time_remaining,
            last_update=status.last_update,
            recent_logs=status.logs[-10:]  # 最近10条日志
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.put("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """取消指定的压缩任务"""
    
    try:
        success = await pipeline.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"任务不存在或无法取消: {job_id}")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "任务取消成功"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, keep_result: bool = True) -> Dict[str, Any]:
    """删除指定的压缩任务"""
    
    try:
        success = pipeline.cleanup_job(job_id, keep_result)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        
        return {
            "job_id": job_id,
            "status": "deleted",
            "message": f"任务删除成功，保留结果: {keep_result}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")

@router.get("/results/{job_id}", response_model=CompressionResultResponse)
async def get_compression_result(job_id: str) -> CompressionResultResponse:
    """获取指定任务的压缩结果"""
    
    try:
        result = await pipeline.get_job_result(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"任务结果不存在: {job_id}")
        
        return CompressionResultResponse(
            job_id=result.job_id,
            compression_ratio=result.compression_ratio,
            speedup_ratio=result.latency_improvement,
            memory_reduction=result.memory_reduction,
            accuracy_retention=1 - result.accuracy_loss if result.accuracy_loss > 0 else None,
            compressed_model_path=result.compressed_model_path,
            evaluation_report_path=result.evaluation_report_path,
            compression_time=result.compression_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取压缩结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取压缩结果失败: {str(e)}")

@router.post("/results/{job_id}/download")
async def download_compressed_model(job_id: str) -> FileResponse:
    """下载压缩后的模型"""
    
    try:
        result = await pipeline.get_job_result(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"任务结果不存在: {job_id}")
        
        model_path = result.compressed_model_path
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"压缩模型文件不存在: {model_path}")
        
        return FileResponse(
            path=model_path,
            filename=f"compressed_model_{job_id}.pth",
            media_type="application/octet-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载压缩模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")

@router.get("/results/{job_id}/report")
async def get_evaluation_report(job_id: str) -> Dict[str, Any]:
    """获取压缩任务的评估报告"""
    
    try:
        result = await pipeline.get_job_result(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"任务结果不存在: {job_id}")
        
        report_path = result.evaluation_report_path
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail=f"评估报告不存在: {report_path}")
        
        import json
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取评估报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取报告失败: {str(e)}")

@router.get("/methods")
async def get_compression_methods() -> Dict[str, Any]:
    """获取支持的压缩方法列表"""
    
    try:
        methods = get_supported_methods()
        return {
            "supported_methods": methods,
            "default_configs": {
                method.value: get_default_config(method) 
                for method in CompressionMethod
            }
        }
    
    except Exception as e:
        logger.error(f"获取压缩方法失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取方法失败: {str(e)}")

@router.post("/methods/validate")
async def validate_compression_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证压缩配置的有效性"""
    
    try:
        compression_method = config.get("compression_method")
        if not compression_method:
            raise HTTPException(status_code=400, detail="缺少compression_method")
        
        # 这里可以添加具体的配置验证逻辑
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 简单验证示例
        if compression_method == "quantization":
            precision = config.get("precision", "int8")
            if precision not in ["fp16", "bf16", "int8", "int4"]:
                validation_result["valid"] = False
                validation_result["errors"].append(f"不支持的精度类型: {precision}")
        
        return validation_result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@router.get("/strategies")
async def get_compression_strategies() -> List[Dict[str, Any]]:
    """获取预定义的压缩策略模板"""
    
    try:
        strategies = [strategy.to_dict() for strategy in DEFAULT_COMPRESSION_STRATEGIES]
        return strategies
    
    except Exception as e:
        logger.error(f"获取压缩策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略失败: {str(e)}")

@router.post("/strategies/recommend")
async def recommend_compression_strategy(request: StrategyRecommendationRequest) -> List[Dict[str, Any]]:
    """根据模型信息推荐压缩策略"""
    
    try:
        from src.ai.model_compression import ModelInfo
        
        model_info = ModelInfo(
            model_name=request.model_name,
            model_type=request.model_type,
            architecture="unknown",  # 可以从model_name推断
            num_parameters=request.num_parameters,
            model_size=request.num_parameters * 4,  # 假设float32
            precision="fp32"
        )
        
        strategies = evaluator.recommend_compression_strategy(
            model_info=model_info,
            target_scenario=request.target_scenario,
            accuracy_tolerance=request.accuracy_tolerance,
            size_reduction_target=request.size_reduction_target
        )
        
        return [strategy.to_dict() for strategy in strategies]
    
    except Exception as e:
        logger.error(f"推荐压缩策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")

@router.post("/benchmark")
async def run_hardware_benchmark(request: BenchmarkRequest) -> List[Dict[str, Any]]:
    """执行硬件性能基准测试"""
    
    try:
        # 验证模型路径
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {request.model_path}")
        
        # 加载模型进行基准测试
        import torch
        model = torch.load(request.model_path, map_location='cpu')
        
        benchmarks = evaluator.benchmark_hardware_performance(
            model=model,
            device_name=request.device_name,
            sequence_lengths=request.sequence_lengths or [128, 256, 512],
            batch_sizes=request.batch_sizes or [1, 2, 4]
        )
        
        return [dict(benchmark) for benchmark in benchmarks]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"硬件基准测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"基准测试失败: {str(e)}")

@router.get("/status")
async def get_pipeline_status() -> Dict[str, Any]:
    """获取压缩流水线的整体状态"""
    
    try:
        active_jobs = pipeline.get_active_jobs()
        queue_length = pipeline.get_job_queue_length()
        
        return {
            "active_jobs": len(active_jobs),
            "queue_length": queue_length,
            "max_concurrent_jobs": pipeline.max_concurrent_jobs,
            "device_info": evaluator.get_device_info(),
            "timestamp": utc_now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"获取流水线状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

# 健康检查端点
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    
    return {
        "status": "healthy",
        "service": "model-compression",
        "timestamp": utc_now().isoformat(),
        "version": "1.0.0"
    }
