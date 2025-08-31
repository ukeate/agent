"""
微调API接口
提供LoRA/QLoRA微调的RESTful API
"""
import os
import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from ...core.database import get_db
from ...ai.fine_tuning import (
    LoRATrainer, QLoRATrainer, DistributedTrainer,
    TrainingConfig, LoRAConfig, QuantizationConfig, 
    ModelArchitecture, TrainingMode, QuantizationType,
    ConfigManager, ModelAdapterFactory, TrainingMonitor
)
from ...services.task_scheduler import TaskScheduler
from ...repositories.task_repository import TaskRepository

# 创建路由
router = APIRouter(prefix="/api/v1/fine-tuning", tags=["微调"])

# 全局变量
config_manager = ConfigManager()
task_scheduler = TaskScheduler()
logger = logging.getLogger(__name__)


# Pydantic模型定义
class LoRAConfigRequest(BaseModel):
    """LoRA配置请求模型"""
    rank: int = Field(16, ge=1, le=512, description="LoRA rank")
    alpha: int = Field(32, ge=1, le=1024, description="LoRA alpha")
    dropout: float = Field(0.1, ge=0, le=1, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(None, description="目标模块列表")
    bias: str = Field("none", description="偏置设置")


class QuantizationConfigRequest(BaseModel):
    """量化配置请求模型"""
    quantization_type: QuantizationType = Field(QuantizationType.NF4, description="量化类型")
    bits: int = Field(4, description="量化位数")
    use_double_quant: bool = Field(True, description="是否使用双重量化")
    quant_type: str = Field("nf4", description="量化方法")
    compute_dtype: str = Field("bfloat16", description="计算数据类型")


class TrainingConfigRequest(BaseModel):
    """训练配置请求模型"""
    job_name: str = Field(..., description="任务名称")
    model_name: str = Field(..., description="模型名称")
    model_architecture: Optional[ModelArchitecture] = Field(None, description="模型架构")
    training_mode: TrainingMode = Field(TrainingMode.LORA, description="训练模式")
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: Optional[str] = Field(None, description="输出目录")
    
    # 训练超参数
    learning_rate: float = Field(2e-4, gt=0, description="学习率")
    num_train_epochs: int = Field(3, ge=1, description="训练轮数")
    per_device_train_batch_size: int = Field(4, ge=1, description="每设备批次大小")
    gradient_accumulation_steps: int = Field(4, ge=1, description="梯度累积步数")
    warmup_steps: int = Field(100, ge=0, description="预热步数")
    max_seq_length: int = Field(2048, ge=64, le=32768, description="最大序列长度")
    
    # LoRA配置
    lora_config: Optional[LoRAConfigRequest] = Field(None, description="LoRA配置")
    
    # 量化配置
    quantization_config: Optional[QuantizationConfigRequest] = Field(None, description="量化配置")
    
    # 分布式配置
    use_distributed: bool = Field(False, description="是否使用分布式训练")
    use_deepspeed: bool = Field(False, description="是否使用DeepSpeed")
    
    # 其他配置
    use_flash_attention: bool = Field(True, description="是否使用Flash Attention")
    use_gradient_checkpointing: bool = Field(True, description="是否使用梯度检查点")
    fp16: bool = Field(False, description="是否使用FP16")
    bf16: bool = Field(True, description="是否使用BF16")


class TrainingJobResponse(BaseModel):
    """训练任务响应模型"""
    job_id: str = Field(..., description="任务ID")
    job_name: str = Field(..., description="任务名称")
    status: str = Field(..., description="任务状态")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    progress: float = Field(0.0, description="进度百分比")
    current_epoch: int = Field(0, description="当前轮次")
    total_epochs: int = Field(0, description="总轮次")
    current_loss: Optional[float] = Field(None, description="当前损失")
    best_loss: Optional[float] = Field(None, description="最佳损失")
    error_message: Optional[str] = Field(None, description="错误信息")


class ModelListResponse(BaseModel):
    """模型列表响应模型"""
    models: List[Dict[str, Any]] = Field(..., description="支持的模型列表")
    architectures: List[str] = Field(..., description="支持的架构列表")


class ConfigTemplateResponse(BaseModel):
    """配置模板响应模型"""
    templates: Dict[str, Dict[str, Any]] = Field(..., description="配置模板")


# API路由实现
@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    config: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """创建微调任务"""
    try:
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 自动检测模型架构（如果未指定）
        if not config.model_architecture:
            config.model_architecture = ModelAdapterFactory.detect_model_architecture(config.model_name)
        
        # 设置输出目录
        if not config.output_dir:
            config.output_dir = f"./fine_tuned_models/{job_id}"
        
        # 创建训练配置
        training_config = _convert_to_training_config(config, job_id)
        
        # 验证配置
        validation_errors = config_manager.validate_config(training_config)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"配置验证失败: {'; '.join(validation_errors)}"
            )
        
        # 保存任务到数据库
        task_repo = TaskRepository(db)
        task_data = {
            "job_id": job_id,
            "job_name": config.job_name,
            "status": "pending",
            "config": training_config.__dict__,
            "created_at": utc_now()
        }
        
        # 将任务添加到后台队列
        background_tasks.add_task(
            _execute_training_job,
            job_id,
            training_config,
            db
        )
        
        logger.info(f"创建微调任务: {job_id}")
        
        return TrainingJobResponse(
            job_id=job_id,
            job_name=config.job_name,
            status="pending",
            created_at=utc_now(),
            current_epoch=0,
            total_epochs=config.num_train_epochs
        )
        
    except Exception as e:
        logger.error(f"创建微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """获取微调任务列表"""
    try:
        task_repo = TaskRepository(db)
        jobs = task_repo.get_jobs(status=status, limit=limit, offset=offset)
        
        return [
            TrainingJobResponse(
                job_id=job["job_id"],
                job_name=job["job_name"],
                status=job["status"],
                created_at=job["created_at"],
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                progress=job.get("progress", 0.0),
                current_epoch=job.get("current_epoch", 0),
                total_epochs=job.get("total_epochs", 0),
                current_loss=job.get("current_loss"),
                best_loss=job.get("best_loss"),
                error_message=job.get("error_message")
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    """获取微调任务详情"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return TrainingJobResponse(**job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str, db: Session = Depends(get_db)):
    """取消微调任务"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job["status"] not in ["pending", "running"]:
            raise HTTPException(status_code=400, detail="任务无法取消")
        
        # 取消任务
        task_scheduler.cancel_task(job_id)
        task_repo.update_job_status(job_id, "cancelled")
        
        return {"message": "任务已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str, db: Session = Depends(get_db)):
    """删除微调任务"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job["status"] == "running":
            raise HTTPException(status_code=400, detail="无法删除运行中的任务")
        
        # 删除任务文件
        if job.get("config", {}).get("output_dir"):
            output_dir = job["config"]["output_dir"]
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
        
        # 删除数据库记录
        task_repo.delete_job(job_id)
        
        return {"message": "任务已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str, 
    lines: int = 100,
    db: Session = Depends(get_db)
):
    """获取训练日志"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 读取日志文件
        log_file = os.path.join(
            job["config"]["output_dir"], 
            "logs", 
            "training_monitor.log"
        )
        
        if not os.path.exists(log_file):
            return {"logs": []}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if lines > 0 else all_lines
        
        return {"logs": [line.strip() for line in recent_lines]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(job_id: str, db: Session = Depends(get_db)):
    """获取训练指标"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 读取指标文件
        metrics_file = os.path.join(
            job["config"]["output_dir"],
            "logs",
            "training_report.json"
        )
        
        if not os.path.exists(metrics_file):
            return {"metrics": {}}
        
        import json
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        return {"metrics": metrics_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/progress")
async def get_training_progress(job_id: str, db: Session = Depends(get_db)):
    """获取训练进度"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 计算进度
        progress_info = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0.0),
            "current_epoch": job.get("current_epoch", 0),
            "total_epochs": job.get("total_epochs", 0),
            "current_loss": job.get("current_loss"),
            "elapsed_time": None,
            "estimated_remaining": None
        }
        
        # 计算耗时
        if job.get("started_at"):
            start_time = job["started_at"]
            current_time = utc_now()
            elapsed = (current_time - start_time).total_seconds()
            progress_info["elapsed_time"] = elapsed
            
            # 估算剩余时间
            if progress_info["progress"] > 0:
                estimated_total = elapsed / (progress_info["progress"] / 100)
                estimated_remaining = estimated_total - elapsed
                progress_info["estimated_remaining"] = max(0, estimated_remaining)
        
        return progress_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练进度失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_training_job(job_id: str, db: Session = Depends(get_db)):
    """暂停训练任务"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job["status"] != "running":
            raise HTTPException(status_code=400, detail="任务未在运行")
        
        # 暂停任务逻辑（需要实现）
        # task_scheduler.pause_task(job_id)
        task_repo.update_job_status(job_id, "paused")
        
        return {"message": "任务已暂停"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_training_job(job_id: str, db: Session = Depends(get_db)):
    """恢复训练任务"""
    try:
        task_repo = TaskRepository(db)
        job = task_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job["status"] != "paused":
            raise HTTPException(status_code=400, detail="任务未暂停")
        
        # 恢复任务逻辑（需要实现）
        # task_scheduler.resume_task(job_id)
        task_repo.update_job_status(job_id, "running")
        
        return {"message": "任务已恢复"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def get_supported_models():
    """获取支持的模型列表"""
    try:
        # 预定义的支持模型列表
        supported_models = {
            "LLaMA": {
                "models": [
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-13b-hf", 
                    "meta-llama/Llama-2-70b-hf",
                    "meta-llama/Meta-Llama-3-8B",
                    "meta-llama/Meta-Llama-3-70B"
                ],
                "architecture": "llama",
                "max_seq_length": 4096
            },
            "Mistral": {
                "models": [
                    "mistralai/Mistral-7B-v0.1",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "mistralai/Mixtral-8x7B-v0.1"
                ],
                "architecture": "mistral",
                "max_seq_length": 8192
            },
            "Qwen": {
                "models": [
                    "Qwen/Qwen-7B",
                    "Qwen/Qwen-14B",
                    "Qwen/Qwen2-7B",
                    "Qwen/Qwen2-72B"
                ],
                "architecture": "qwen",
                "max_seq_length": 8192
            },
            "ChatGLM": {
                "models": [
                    "THUDM/chatglm3-6b",
                    "THUDM/chatglm3-6b-base"
                ],
                "architecture": "chatglm",
                "max_seq_length": 8192
            }
        }
        
        return ModelListResponse(
            models=list(supported_models.values()),
            architectures=ModelAdapterFactory.get_supported_architectures()
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/validate")
async def validate_model_config(config: TrainingConfigRequest):
    """验证模型配置"""
    try:
        # 创建模型适配器
        adapter = ModelAdapterFactory.create_adapter(config.model_name)
        
        # 检测硬件配置
        hardware_config = config_manager.detect_hardware_config()
        
        # 获取模型特定配置
        model_config = config_manager.get_model_specific_config(config.model_name)
        
        # 验证配置兼容性
        from ...ai.fine_tuning.model_adapters import ModelOptimizer
        optimizer = ModelOptimizer(adapter)
        warnings = optimizer.validate_configuration(config.dict())
        
        # 推荐配置
        recommendations = {}
        if hardware_config["hardware_info"]["cuda_available"]:
            gpu_memory = hardware_config["hardware_info"]["total_memory"][0] if hardware_config["hardware_info"]["total_memory"] else 8
            batch_size, grad_acc = optimizer.get_recommended_batch_size(gpu_memory, config.max_seq_length)
            recommendations["batch_size"] = batch_size
            recommendations["gradient_accumulation_steps"] = grad_acc
            recommendations["quantization"] = optimizer.get_quantization_recommendations(gpu_memory)
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "model_info": {
                "architecture": adapter.get_architecture().value,
                "max_seq_length": adapter.get_max_sequence_length(),
                "target_modules": adapter.get_target_modules()
            },
            "hardware_info": hardware_config,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"验证模型配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs/templates", response_model=ConfigTemplateResponse)
async def get_config_templates():
    """获取配置模板"""
    try:
        templates = {}
        for template_name in config_manager.list_templates():
            templates[template_name] = config_manager.get_template(template_name)
        
        return ConfigTemplateResponse(templates=templates)
        
    except Exception as e:
        logger.error(f"获取配置模板失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/validate")
async def validate_training_config(config: TrainingConfigRequest):
    """验证训练配置"""
    try:
        # 转换为内部配置格式
        training_config = _convert_to_training_config(config, "validation")
        
        # 验证配置
        validation_errors = config_manager.validate_config(training_config)
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors
        }
        
    except Exception as e:
        logger.error(f"验证训练配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """上传训练数据集"""
    try:
        # 验证文件类型
        if not file.filename.endswith(('.json', '.jsonl', '.csv')):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 创建数据集目录
        datasets_dir = "./datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(datasets_dir, f"{name}_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 验证数据集格式
        dataset_info = _validate_dataset_format(file_path)
        
        return {
            "message": "数据集上传成功",
            "file_path": file_path,
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        logger.error(f"上传数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    """获取数据集列表"""
    try:
        datasets_dir = "./datasets"
        if not os.path.exists(datasets_dir):
            return {"datasets": []}
        
        datasets = []
        for filename in os.listdir(datasets_dir):
            file_path = os.path.join(datasets_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                datasets.append({
                    "filename": filename,
                    "path": file_path,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime)
                })
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """获取数据集详情"""
    try:
        # 这里简化处理，实际应该有数据集ID映射
        datasets_dir = "./datasets"
        file_path = os.path.join(datasets_dir, dataset_id)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="数据集不存在")
        
        dataset_info = _validate_dataset_format(file_path)
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset_format(dataset_id: str):
    """验证数据集格式"""
    try:
        datasets_dir = "./datasets"
        file_path = os.path.join(datasets_dir, dataset_id)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="数据集不存在")
        
        validation_result = _validate_dataset_format(file_path)
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证数据集格式失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
def _convert_to_training_config(config: TrainingConfigRequest, job_id: str) -> TrainingConfig:
    """将请求配置转换为训练配置"""
    
    # 转换LoRA配置
    lora_config = None
    if config.lora_config:
        lora_config = LoRAConfig(
            rank=config.lora_config.rank,
            alpha=config.lora_config.alpha,
            dropout=config.lora_config.dropout,
            target_modules=config.lora_config.target_modules,
            bias=config.lora_config.bias
        )
    
    # 转换量化配置
    quantization_config = None
    if config.quantization_config:
        quantization_config = QuantizationConfig(
            quantization_type=config.quantization_config.quantization_type,
            bits=config.quantization_config.bits,
            use_double_quant=config.quantization_config.use_double_quant,
            quant_type=config.quantization_config.quant_type,
            compute_dtype=config.quantization_config.compute_dtype
        )
    
    # 设置输出目录
    output_dir = config.output_dir or f"./fine_tuned_models/{job_id}"
    
    return TrainingConfig(
        model_name=config.model_name,
        model_architecture=config.model_architecture or ModelAdapterFactory.detect_model_architecture(config.model_name),
        training_mode=config.training_mode,
        dataset_path=config.dataset_path,
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_seq_length=config.max_seq_length,
        lora_config=lora_config,
        quantization_config=quantization_config,
        use_distributed=config.use_distributed,
        use_deepspeed=config.use_deepspeed,
        use_flash_attention=config.use_flash_attention,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        fp16=config.fp16,
        bf16=config.bf16
    )


async def _execute_training_job(job_id: str, config: TrainingConfig, db: Session):
    """执行训练任务"""
    task_repo = TaskRepository(db)
    
    try:
        # 更新任务状态
        task_repo.update_job_status(job_id, "running")
        task_repo.update_job_started_at(job_id, utc_now())
        
        # 创建训练监控器
        monitor = TrainingMonitor(
            log_dir=os.path.join(config.output_dir, "logs"),
            enable_wandb=False  # 可以根据需要启用
        )
        
        # 根据配置创建相应的训练器
        if config.use_distributed:
            trainer = DistributedTrainer(config, monitor)
        elif config.training_mode == TrainingMode.QLORA:
            trainer = QLoRATrainer(config, monitor)
        else:
            trainer = LoRATrainer(config, monitor)
        
        # 执行训练
        with monitor:
            result = trainer.train()
        
        # 更新任务状态为完成
        task_repo.update_job_status(job_id, "completed")
        task_repo.update_job_completed_at(job_id, utc_now())
        task_repo.update_job_result(job_id, result)
        
        logger.info(f"训练任务 {job_id} 完成")
        
    except Exception as e:
        logger.error(f"训练任务 {job_id} 失败: {str(e)}")
        
        # 更新任务状态为失败
        task_repo.update_job_status(job_id, "failed")
        task_repo.update_job_error(job_id, str(e))


def _validate_dataset_format(file_path: str) -> Dict[str, Any]:
    """验证数据集格式"""
    import json
    
    try:
        dataset_info = {
            "format": "unknown",
            "samples": 0,
            "valid": False,
            "errors": [],
            "sample_data": None
        }
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                dataset_info["format"] = "json_list"
                dataset_info["samples"] = len(data)
                
                # 检查必需字段
                if data and isinstance(data[0], dict):
                    sample = data[0]
                    if "instruction" in sample and "output" in sample:
                        dataset_info["valid"] = True
                        dataset_info["sample_data"] = sample
                    else:
                        dataset_info["errors"].append("缺少必需字段: instruction, output")
                else:
                    dataset_info["errors"].append("数据格式错误")
                    
        elif file_path.endswith('.jsonl'):
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        samples.append(data)
                        if line_num > 1000:  # 只读取前1000行进行验证
                            break
                    except json.JSONDecodeError as e:
                        dataset_info["errors"].append(f"第{line_num}行JSON格式错误: {str(e)}")
            
            dataset_info["format"] = "jsonl"
            dataset_info["samples"] = len(samples)
            
            if samples:
                sample = samples[0]
                if "instruction" in sample and "output" in sample:
                    dataset_info["valid"] = True
                    dataset_info["sample_data"] = sample
                else:
                    dataset_info["errors"].append("缺少必需字段: instruction, output")
        
        return dataset_info
        
    except Exception as e:
        return {
            "format": "unknown",
            "samples": 0,
            "valid": False,
            "errors": [f"读取文件失败: {str(e)}"],
            "sample_data": None
        }