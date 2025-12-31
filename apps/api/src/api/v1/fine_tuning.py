"""
微调API接口
提供LoRA/QLoRA微调的RESTful API
"""

import os
import json
import uuid
import asyncio
import threading
import re
import ast
import tempfile
import zipfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from pydantic import Field
from src.ai.fine_tuning import (
    LoRATrainer, QLoRATrainer, DistributedTrainer,
    TrainingConfig, LoRAConfig, QuantizationConfig, 
    ModelArchitecture, TrainingMode, QuantizationType,
    ConfigManager, ModelAdapterFactory, TrainingMonitor
)

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/fine-tuning", tags=["微调"])

# 全局变量
config_manager = ConfigManager()

_JOBS_DIR = "./fine_tuning_jobs"
_job_lock = threading.Lock()
_job_controls: Dict[str, "JobControl"] = {}

class JobControl:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()

def _ensure_jobs_dir() -> None:
    os.makedirs(_JOBS_DIR, exist_ok=True)

def _job_path(job_id: str) -> str:
    return os.path.join(_JOBS_DIR, f"{job_id}.json")

def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        job = json.load(f)
    normalized = _normalize_job_config(job)
    if normalized:
        _write_job(job)
    return job

def _parse_structured_config(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or "(" not in raw or ")" not in raw:
        return None
    body = raw[raw.find("(") + 1:raw.rfind(")")]
    if not body:
        return None
    result: Dict[str, Any] = {}
    for part in [p.strip() for p in body.split(",") if p.strip()]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value in ("None", "null"):
            parsed = None
        elif value in ("True", "False"):
            parsed = value == "True"
        elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
            parsed = value[1:-1]
        elif value.startswith("[") or value.startswith("{") or value.startswith("("):
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                parsed = value
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        result[key] = parsed
    return result or None

def _normalize_job_config(job: Dict[str, Any]) -> bool:
    config = job.get("config")
    if not isinstance(config, dict):
        return False
    updated = False
    lora_config = config.get("lora_config")
    if isinstance(lora_config, str):
        if lora_config in ("None", "null"):
            config["lora_config"] = None
            updated = True
        else:
            parsed = _parse_structured_config(lora_config)
            if parsed is not None:
                config["lora_config"] = parsed
                updated = True
    quant_config = config.get("quantization_config")
    if isinstance(quant_config, str):
        if quant_config in ("None", "null"):
            config["quantization_config"] = None
            updated = True
        else:
            parsed = _parse_structured_config(quant_config)
            if parsed is not None:
                config["quantization_config"] = parsed
                updated = True
    if updated:
        job["config"] = config
    return updated

def _write_job(job: Dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("job_id不能为空")
    _ensure_jobs_dir()
    path = _job_path(job_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2, default=str)

def _update_job(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    with _job_lock:
        job = _read_job(job_id)
        if not job:
            raise KeyError("任务不存在")
        job.update(updates)
        _write_job(job)
        return job

def _list_jobs() -> List[Dict[str, Any]]:
    _ensure_jobs_dir()
    jobs: List[Dict[str, Any]] = []
    for name in os.listdir(_JOBS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            job_id = name.removesuffix(".json")
            job = _read_job(job_id)
            if job:
                jobs.append(job)
        except Exception:
            continue
    jobs.sort(key=lambda j: str(j.get("created_at") or ""), reverse=True)
    return jobs

def _get_control(job_id: str) -> JobControl:
    with _job_lock:
        ctrl = _job_controls.get(job_id)
        if ctrl:
            return ctrl
        ctrl = JobControl()
        _job_controls[job_id] = ctrl
        return ctrl

# Pydantic模型定义
class LoRAConfigRequest(ApiBaseModel):
    """LoRA配置请求模型"""
    rank: int = Field(16, ge=1, le=512, description="LoRA rank")
    alpha: int = Field(32, ge=1, le=1024, description="LoRA alpha")
    dropout: float = Field(0.1, ge=0, le=1, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(None, description="目标模块列表")
    bias: str = Field("none", description="偏置设置")

class QuantizationConfigRequest(ApiBaseModel):
    """量化配置请求模型"""
    quantization_type: QuantizationType = Field(QuantizationType.NF4, description="量化类型")
    bits: int = Field(4, description="量化位数")
    use_double_quant: bool = Field(True, description="是否使用双重量化")
    quant_type: str = Field("nf4", description="量化方法")
    compute_dtype: str = Field("bfloat16", description="计算数据类型")

class TrainingConfigRequest(ApiBaseModel):
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

class TrainingJobResponse(ApiBaseModel):
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
    config: Optional[Dict[str, Any]] = Field(None, description="训练配置")

class ModelListResponse(ApiBaseModel):
    """模型列表响应模型"""
    models: List[Dict[str, Any]] = Field(..., description="支持的模型列表")
    architectures: List[str] = Field(..., description="支持的架构列表")

class ConfigTemplateResponse(ApiBaseModel):
    """配置模板响应模型"""
    templates: Dict[str, Dict[str, Any]] = Field(..., description="配置模板")

# API路由实现
@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    config: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
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

        now = utc_now()
        os.makedirs(os.path.join(training_config.output_dir, "logs"), exist_ok=True)
        _write_job(
            {
                "job_id": job_id,
                "job_name": config.job_name,
                "status": "pending",
                "created_at": now.isoformat(),
                "started_at": None,
                "completed_at": None,
                "progress": 0.0,
                "current_epoch": 0,
                "total_epochs": int(config.num_train_epochs),
                "current_loss": None,
                "best_loss": None,
                "error_message": None,
                "config": training_config.__dict__,
            }
        )
        _get_control(job_id)
        background_tasks.add_task(_execute_training_job, job_id, training_config)
        
        logger.info(f"创建微调任务: {job_id}")
        
        return TrainingJobResponse(
            job_id=job_id,
            job_name=config.job_name,
            status="pending",
            created_at=now,
            progress=0.0,
            current_epoch=0,
            total_epochs=config.num_train_epochs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建微调任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """获取微调任务列表"""
    try:
        jobs = _list_jobs()
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        if offset > 0:
            jobs = jobs[offset:]
        if limit > 0:
            jobs = jobs[:limit]
        
        return [
            TrainingJobResponse(
                job_id=job["job_id"],
                job_name=job["job_name"],
                status=job["status"],
                created_at=job["created_at"],
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                progress=float(job.get("progress") or 0.0),
                current_epoch=int(job.get("current_epoch") or 0),
                total_epochs=int(job.get("total_epochs") or 0),
                current_loss=job.get("current_loss"),
                best_loss=job.get("best_loss"),
                error_message=job.get("error_message"),
                config=job.get("config")
            )
            for job in jobs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """获取微调任务详情"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return TrainingJobResponse(**job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """取消微调任务"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job.get("status") not in ["pending", "running", "paused"]:
            raise HTTPException(status_code=400, detail="任务无法取消")
        
        ctrl = _get_control(job_id)
        ctrl.pause_event.clear()
        ctrl.cancel_event.set()
        _update_job(job_id, {"status": "cancelled", "completed_at": utc_now().isoformat()})
        
        return {"message": "任务已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """删除微调任务"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job.get("status") == "running":
            raise HTTPException(status_code=400, detail="无法删除运行中的任务")
        
        # 删除任务文件
        if job.get("config", {}).get("output_dir"):
            output_dir = job["config"]["output_dir"]
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
        
        with _job_lock:
            path = _job_path(job_id)
            if os.path.exists(path):
                os.remove(path)
            _job_controls.pop(job_id, None)
        
        return {"message": "任务已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/download")
async def download_training_artifacts(job_id: str, background_tasks: BackgroundTasks):
    """下载训练产物"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        output_dir = job.get("config", {}).get("output_dir")
        if not output_dir or not os.path.isdir(output_dir):
            raise HTTPException(status_code=404, detail="产物目录不存在")
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp_path = tmp.name
        tmp.close()
        
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    arcname = os.path.relpath(full_path, output_dir)
                    zf.write(full_path, arcname=arcname)
        
        background_tasks.add_task(os.remove, tmp_path)
        return FileResponse(tmp_path, filename=f"{job_id}.zip", media_type="application/zip")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载产物失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str, 
    lines: int = 100,
):
    """获取训练日志"""
    try:
        job = _read_job(job_id)
        
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
async def get_training_metrics(job_id: str):
    """获取训练指标"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        logs_dir = os.path.join(job["config"]["output_dir"], "logs")
        if not os.path.isdir(logs_dir):
            return {"metrics": {}}
        
        candidates = [
            os.path.join(logs_dir, name)
            for name in os.listdir(logs_dir)
            if name.startswith("training_report_") and name.endswith(".json")
        ]
        if not candidates:
            return {"metrics": {}}
        metrics_file = max(candidates, key=os.path.getmtime)
        
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        return {"metrics": metrics_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/progress")
async def get_training_progress(job_id: str):
    """获取训练进度"""
    try:
        job = _read_job(job_id)
        
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
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
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
async def pause_training_job(job_id: str):
    """暂停训练任务"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job.get("status") != "running":
            raise HTTPException(status_code=400, detail="任务未在运行")
        
        ctrl = _get_control(job_id)
        ctrl.pause_event.set()
        _update_job(job_id, {"status": "paused"})
        
        return {"message": "任务已暂停"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/{job_id}/resume")
async def resume_training_job(job_id: str):
    """恢复训练任务"""
    try:
        job = _read_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if job.get("status") != "paused":
            raise HTTPException(status_code=400, detail="任务未暂停")
        
        ctrl = _get_control(job_id)
        ctrl.pause_event.clear()
        _update_job(job_id, {"status": "running"})
        
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
        cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
        by_arch: Dict[str, Dict[str, Any]] = {}
        if os.path.isdir(cache_dir):
            for name in os.listdir(cache_dir):
                if not name.startswith("models--"):
                    continue
                raw = name.removeprefix("models--")
                parts = raw.split("--")
                if len(parts) < 2:
                    continue
                repo_id = f"{parts[0]}/{'--'.join(parts[1:])}"
                if not any(re.search(p, repo_id.lower()) for p in ModelAdapterFactory.ADAPTER_MAPPING):
                    continue
                try:
                    adapter = ModelAdapterFactory.create_adapter(repo_id)
                    arch = adapter.get_architecture().value
                    max_len = adapter.get_max_sequence_length()
                except Exception:
                    continue
                group = by_arch.get(arch)
                if not group:
                    group = {"models": [], "architecture": arch, "max_seq_length": max_len}
                    by_arch[arch] = group
                if repo_id not in group["models"]:
                    group["models"].append(repo_id)
                    group["models"].sort()
                group["max_seq_length"] = max(group["max_seq_length"], max_len)
        
        return ModelListResponse(
            models=[by_arch[k] for k in sorted(by_arch)],
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
        from src.ai.fine_tuning.model_adapters import ModelOptimizer
        optimizer = ModelOptimizer(adapter)
        warnings = optimizer.validate_configuration(config.model_dump())
        
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

async def _execute_training_job(job_id: str, config: TrainingConfig):
    """执行训练任务"""
    ctrl = _get_control(job_id)
    try:
        _update_job(job_id, {"status": "running", "started_at": utc_now().isoformat()})
        
        with TrainingMonitor(log_dir=os.path.join(config.output_dir, "logs"), enable_wandb=False) as monitor:
            if config.use_distributed:
                trainer = DistributedTrainer(config, monitor)
            elif config.training_mode == TrainingMode.QLORA:
                trainer = QLoRATrainer(config, monitor, job_control=ctrl)
            else:
                trainer = LoRATrainer(config, monitor, job_control=ctrl)
            
            result = await asyncio.to_thread(trainer.train)
        if ctrl.cancel_event.is_set():
            _update_job(job_id, {"status": "cancelled", "completed_at": utc_now().isoformat()})
            return
        
        # 更新任务状态为完成
        _update_job(
            job_id,
            {
                "status": "completed",
                "completed_at": utc_now().isoformat(),
                "progress": 100.0,
                "current_epoch": int(getattr(config, "num_train_epochs", 0) or 0),
                "total_epochs": int(getattr(config, "num_train_epochs", 0) or 0),
                "result": result,
            },
        )
        
        logger.info(f"训练任务 {job_id} 完成")
        
    except Exception as e:
        logger.error(f"训练任务 {job_id} 失败: {str(e)}")
        if ctrl.cancel_event.is_set():
            _update_job(job_id, {"status": "cancelled", "completed_at": utc_now().isoformat()})
            return
        _update_job(
            job_id,
            {
                "status": "failed",
                "completed_at": utc_now().isoformat(),
                "error_message": str(e),
            },
        )

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
