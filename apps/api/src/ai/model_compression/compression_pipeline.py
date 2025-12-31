"""
压缩流水线管理

实现压缩策略组合和自动化
添加压缩任务调度和监控
实现压缩结果版本管理
提供压缩配置模板和最佳实践
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import time
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
from .models import (
    CompressionJob,
    CompressionResult,
    CompressionMethod,
    QuantizationConfig,
    DistillationConfig,
    PruningConfig,
    CompressionStrategy,
    ModelInfo
)
from .quantization_engine import QuantizationEngine
from .distillation_trainer import DistillationTrainer
from .pruning_engine import PruningEngine
from .compression_evaluator import CompressionEvaluator

from src.core.logging import get_logger
@dataclass
class PipelineStatus:
    """流水线状态"""
    job_id: str
    current_stage: str
    progress_percent: float
    estimated_time_remaining: float  # seconds
    last_update: datetime
    logs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "current_stage": self.current_stage,
            "progress_percent": self.progress_percent,
            "estimated_time_remaining": self.estimated_time_remaining,
            "last_update": self.last_update.isoformat(),
            "recent_logs": self.logs[-10:]  # 只返回最近10条日志
        }

class CompressionPipeline:
    """模型压缩流水线
    
    统一管理整个压缩流程:
    - 任务队列和调度
    - 多种压缩方法的协调执行
    - 中间结果保存和恢复
    - 性能监控和日志记录
    - 错误处理和恢复
    """
    
    def __init__(self, 
                 workspace_dir: str = "compression_workspace",
                 max_concurrent_jobs: int = 2):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_jobs = max_concurrent_jobs
        self.logger = get_logger(__name__)
        
        # 初始化压缩引擎
        self.quantization_engine = QuantizationEngine()
        self.compression_evaluator = CompressionEvaluator()
        
        # 任务管理
        self.active_jobs = {}  # job_id -> job_info
        self.job_queue = []
        self.job_history = {}
        self.job_lock = threading.Lock()
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        # 状态监控
        self.job_status = {}  # job_id -> PipelineStatus
        
        # 压缩阶段定义
        self.compression_stages = {
            CompressionMethod.QUANTIZATION: [
                ("load_model", "加载模型"),
                ("prepare_calibration", "准备校准数据"),
                ("quantize", "执行量化"),
                ("evaluate", "性能评估"),
                ("save_result", "保存结果")
            ],
            CompressionMethod.DISTILLATION: [
                ("load_models", "加载师生模型"),
                ("prepare_training", "准备训练数据"),
                ("distill", "执行知识蒸馏"),
                ("evaluate", "性能评估"),
                ("save_result", "保存结果")
            ],
            CompressionMethod.PRUNING: [
                ("load_model", "加载模型"),
                ("analyze_structure", "分析模型结构"),
                ("prune", "执行剪枝"),
                ("recovery_training", "恢复训练"),
                ("evaluate", "性能评估"),
                ("save_result", "保存结果")
            ],
            CompressionMethod.MIXED: [
                ("load_model", "加载模型"),
                ("stage1_compression", "第一阶段压缩"),
                ("stage2_compression", "第二阶段压缩"),
                ("final_optimization", "最终优化"),
                ("evaluate", "性能评估"),
                ("save_result", "保存结果")
            ]
        }
    
    async def submit_job(self, job: CompressionJob) -> str:
        """提交压缩任务"""
        
        self.logger.info(f"提交压缩任务: {job.job_id}")
        
        # 验证任务配置
        if not self._validate_job(job):
            raise ValueError("任务配置无效")
        
        # 创建工作目录
        job_dir = self.workspace_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存任务配置
        self._save_job_config(job, job_dir)
        
        # 初始化任务状态
        stages = self.compression_stages.get(job.compression_method, [])
        status = PipelineStatus(
            job_id=job.job_id,
            current_stage="queued",
            progress_percent=0.0,
            estimated_time_remaining=0.0,
            last_update=utc_now(),
            logs=[f"任务 {job.job_id} 已提交到队列"]
        )
        
        with self.job_lock:
            self.job_queue.append(job)
            self.job_status[job.job_id] = status
            job.status = "queued"
        
        # 尝试启动任务执行
        self._try_start_next_job()
        
        return job.job_id
    
    async def get_job_status(self, job_id: str) -> Optional[PipelineStatus]:
        """获取任务状态"""
        
        with self.job_lock:
            return self.job_status.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        
        self.logger.info(f"取消任务: {job_id}")
        
        with self.job_lock:
            # 从队列中移除
            self.job_queue = [job for job in self.job_queue if job.job_id != job_id]
            
            # 标记为取消
            if job_id in self.job_status:
                self.job_status[job_id].current_stage = "cancelled"
                self.job_status[job_id].logs.append("任务已取消")
                self.job_status[job_id].last_update = utc_now()
            
            # 如果任务正在运行，尝试停止
            if job_id in self.active_jobs:
                # 这里可以添加更复杂的任务中断逻辑
                self.active_jobs[job_id]["cancelled"] = True
        
        return True
    
    async def get_job_result(self, job_id: str) -> Optional[CompressionResult]:
        """获取任务结果"""
        
        job_dir = self.workspace_dir / job_id
        result_file = job_dir / "result.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    return CompressionResult(**result_data)
            except Exception as e:
                self.logger.error(f"加载任务结果失败 {job_id}: {e}")
        
        return None
    
    def _try_start_next_job(self) -> None:
        """尝试启动下一个任务"""
        
        with self.job_lock:
            if len(self.active_jobs) >= self.max_concurrent_jobs:
                return
            
            if not self.job_queue:
                return
            
            # 取出队列中的第一个任务
            job = self.job_queue.pop(0)
            self.active_jobs[job.job_id] = {
                "job": job,
                "started_at": utc_now(),
                "cancelled": False
            }
        
        # 异步启动任务执行
        self.executor.submit(self._execute_job, job)
    
    def _execute_job(self, job: CompressionJob) -> None:
        """执行压缩任务"""
        
        try:
            self.logger.info(f"开始执行任务: {job.job_id}")
            
            # 更新状态
            self._update_job_status(job.job_id, "running", 0.0, "开始执行压缩任务")
            
            # 根据压缩方法执行相应的流程
            if job.compression_method == CompressionMethod.QUANTIZATION:
                result = self._execute_quantization(job)
            elif job.compression_method == CompressionMethod.DISTILLATION:
                result = self._execute_distillation(job)
            elif job.compression_method == CompressionMethod.PRUNING:
                result = self._execute_pruning(job)
            elif job.compression_method == CompressionMethod.MIXED:
                result = self._execute_mixed_compression(job)
            else:
                raise ValueError(f"不支持的压缩方法: {job.compression_method}")
            
            # 保存结果
            self._save_job_result(job.job_id, result)
            
            # 更新状态为完成
            self._update_job_status(job.job_id, "completed", 100.0, "任务执行完成")
            job.status = "completed"
            job.completed_at = utc_now()
            
            self.logger.info(f"任务执行完成: {job.job_id}")
            
        except Exception as e:
            self.logger.error(f"任务执行失败 {job.job_id}: {e}")
            
            # 更新状态为失败
            self._update_job_status(job.job_id, "failed", 0.0, f"任务执行失败: {str(e)}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = utc_now()
        
        finally:
            # 清理活跃任务
            with self.job_lock:
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
            
            # 尝试启动下一个任务
            self._try_start_next_job()
    
    def _execute_quantization(self, job: CompressionJob) -> CompressionResult:
        """执行量化压缩"""
        
        if not job.quantization_config:
            raise ValueError("量化配置缺失")
        
        self._update_job_status(job.job_id, "load_model", 10.0, "加载模型")
        
        # 加载模型
        model = self._load_model(job.model_path)
        original_size = self._get_model_size(model)
        original_params = sum(p.numel() for p in model.parameters())
        
        self._update_job_status(job.job_id, "quantize", 30.0, "执行量化")
        
        # 准备校准数据（如果需要）
        calibration_data = None
        if job.quantization_config.calibration_dataset_size > 0:
            calibration_data = self._prepare_calibration_data(job)
        
        # 执行量化
        quantized_model, quantization_stats = self.quantization_engine.quantize_model(
            model, job.quantization_config, calibration_data
        )
        
        self._update_job_status(job.job_id, "evaluate", 70.0, "评估性能")
        
        # 性能评估
        comparison_report = self.compression_evaluator.compare_models(
            model, quantized_model
        )
        
        self._update_job_status(job.job_id, "save_result", 90.0, "保存结果")
        
        # 保存量化模型
        job_dir = self.workspace_dir / job.job_id
        compressed_model_path = str(job_dir / "quantized_model.pth")
        self.quantization_engine.save_quantized_model(quantized_model, compressed_model_path)
        
        # 创建结果对象
        result = CompressionResult(
            job_id=job.job_id,
            original_model_size=original_size,
            compressed_model_size=self._get_model_size(quantized_model),
            compression_ratio=quantization_stats["compression_ratio"],
            original_params=original_params,
            compressed_params=sum(p.numel() for p in quantized_model.parameters()),
            param_reduction_ratio=quantization_stats["param_reduction"],
            compressed_model_path=compressed_model_path,
            evaluation_report_path=str(job_dir / "evaluation_report.json"),
            compression_method=job.compression_method,
            compression_config=quantization_stats["quantization_info"],
            compression_time=quantization_stats["compression_time"]
        )
        
        # 保存评估报告
        evaluation_report = self.compression_evaluator.generate_evaluation_report(
            comparison_report, result.evaluation_report_path
        )
        
        return result
    
    def _execute_distillation(self, job: CompressionJob) -> CompressionResult:
        """执行知识蒸馏"""
        
        if not job.distillation_config:
            raise ValueError("蒸馏配置缺失")
        
        self._update_job_status(job.job_id, "load_models", 10.0, "加载师生模型")
        
        # 创建蒸馏训练器
        trainer = DistillationTrainer(job.distillation_config)
        trainer.load_models()
        
        original_params = trainer._count_parameters(trainer.teacher_model)
        
        self._update_job_status(job.job_id, "prepare_training", 20.0, "准备训练数据")
        
        # 准备训练数据
        train_dataloader = self._prepare_training_data(job)
        
        self._update_job_status(job.job_id, "distill", 40.0, "执行知识蒸馏")
        
        # 执行蒸馏
        distillation_result = trainer.distill(train_dataloader)
        
        self._update_job_status(job.job_id, "evaluate", 80.0, "评估性能")
        
        # 性能评估
        comparison_report = self.compression_evaluator.compare_models(
            trainer.teacher_model, distillation_result.student_model
        )
        
        self._update_job_status(job.job_id, "save_result", 95.0, "保存结果")
        
        # 保存学生模型
        job_dir = self.workspace_dir / job.job_id
        compressed_model_path = str(job_dir / "student_model.pth")
        trainer.save_student_model(compressed_model_path)
        
        # 创建结果对象
        result = CompressionResult(
            job_id=job.job_id,
            original_model_size=self._get_model_size(trainer.teacher_model),
            compressed_model_size=self._get_model_size(distillation_result.student_model),
            compression_ratio=trainer.get_compression_ratio(),
            original_params=original_params,
            compressed_params=trainer._count_parameters(distillation_result.student_model),
            param_reduction_ratio=trainer.get_parameter_reduction(),
            compressed_model_path=compressed_model_path,
            evaluation_report_path=str(job_dir / "evaluation_report.json"),
            compression_method=job.compression_method,
            compression_config=job.distillation_config.to_dict(),
            compression_time=distillation_result.training_time
        )
        
        # 保存评估报告
        evaluation_report = self.compression_evaluator.generate_evaluation_report(
            comparison_report, result.evaluation_report_path
        )
        
        return result
    
    def _execute_pruning(self, job: CompressionJob) -> CompressionResult:
        """执行模型剪枝"""
        
        if not job.pruning_config:
            raise ValueError("剪枝配置缺失")
        
        self._update_job_status(job.job_id, "load_model", 10.0, "加载模型")
        
        # 加载模型
        model = self._load_model(job.model_path)
        original_size = self._get_model_size(model)
        
        self._update_job_status(job.job_id, "prune", 30.0, "执行剪枝")
        
        # 创建剪枝引擎
        pruning_engine = PruningEngine(job.pruning_config)
        
        # 准备训练数据（用于恢复训练）
        train_dataloader = None
        if job.pruning_config.gradual_pruning and job.pruning_config.recovery_epochs > 0:
            train_dataloader = self._prepare_training_data(job)
        
        # 执行剪枝
        pruned_model, pruning_result = pruning_engine.prune_model(model, train_dataloader)
        
        self._update_job_status(job.job_id, "evaluate", 70.0, "评估性能")
        
        # 性能评估
        comparison_report = self.compression_evaluator.compare_models(model, pruned_model)
        
        self._update_job_status(job.job_id, "save_result", 90.0, "保存结果")
        
        # 保存剪枝模型
        job_dir = self.workspace_dir / job.job_id
        compressed_model_path = str(job_dir / "pruned_model.pth")
        torch.save(pruned_model, compressed_model_path)
        
        # 创建结果对象
        result = CompressionResult(
            job_id=job.job_id,
            original_model_size=original_size,
            compressed_model_size=self._get_model_size(pruned_model),
            compression_ratio=original_size / self._get_model_size(pruned_model),
            original_params=pruning_result.original_params,
            compressed_params=pruning_result.pruned_params,
            param_reduction_ratio=pruning_result.param_reduction,
            compressed_model_path=compressed_model_path,
            evaluation_report_path=str(job_dir / "evaluation_report.json"),
            compression_method=job.compression_method,
            compression_config=job.pruning_config.to_dict(),
            compression_time=pruning_result.pruning_time
        )
        
        # 保存评估报告
        evaluation_report = self.compression_evaluator.generate_evaluation_report(
            comparison_report, result.evaluation_report_path
        )
        
        return result
    
    def _execute_mixed_compression(self, job: CompressionJob) -> CompressionResult:
        """执行混合压缩（多种方法组合）"""
        
        self._update_job_status(job.job_id, "load_model", 5.0, "加载模型")
        
        # 加载模型
        model = self._load_model(job.model_path)
        original_size = self._get_model_size(model)
        original_params = sum(p.numel() for p in model.parameters())
        
        current_model = model
        compression_configs = []
        total_compression_time = 0.0
        
        # 第一阶段：剪枝（如果配置了）
        if job.pruning_config:
            self._update_job_status(job.job_id, "stage1_compression", 25.0, "执行剪枝")
            
            pruning_engine = PruningEngine(job.pruning_config)
            train_dataloader = self._prepare_training_data(job)
            
            current_model, pruning_result = pruning_engine.prune_model(
                current_model, train_dataloader
            )
            
            compression_configs.append({"pruning": job.pruning_config.to_dict()})
            total_compression_time += pruning_result.pruning_time
        
        # 第二阶段：量化（如果配置了）
        if job.quantization_config:
            self._update_job_status(job.job_id, "stage2_compression", 60.0, "执行量化")
            
            calibration_data = self._prepare_calibration_data(job)
            current_model, quantization_stats = self.quantization_engine.quantize_model(
                current_model, job.quantization_config, calibration_data
            )
            
            compression_configs.append({"quantization": quantization_stats["quantization_info"]})
            total_compression_time += quantization_stats["compression_time"]
        
        # 最终优化（如果需要）
        self._update_job_status(job.job_id, "final_optimization", 80.0, "最终优化")
        
        # 这里可以添加额外的优化步骤，比如模型融合、算子优化等
        
        self._update_job_status(job.job_id, "evaluate", 90.0, "评估性能")
        
        # 性能评估
        comparison_report = self.compression_evaluator.compare_models(model, current_model)
        
        self._update_job_status(job.job_id, "save_result", 95.0, "保存结果")
        
        # 保存压缩模型
        job_dir = self.workspace_dir / job.job_id
        compressed_model_path = str(job_dir / "mixed_compressed_model.pth")
        torch.save(current_model, compressed_model_path)
        
        # 创建结果对象
        result = CompressionResult(
            job_id=job.job_id,
            original_model_size=original_size,
            compressed_model_size=self._get_model_size(current_model),
            compression_ratio=original_size / self._get_model_size(current_model),
            original_params=original_params,
            compressed_params=sum(p.numel() for p in current_model.parameters()),
            param_reduction_ratio=(original_params - sum(p.numel() for p in current_model.parameters())) / original_params,
            compressed_model_path=compressed_model_path,
            evaluation_report_path=str(job_dir / "evaluation_report.json"),
            compression_method=job.compression_method,
            compression_config={"stages": compression_configs},
            compression_time=total_compression_time
        )
        
        # 保存评估报告
        evaluation_report = self.compression_evaluator.generate_evaluation_report(
            comparison_report, result.evaluation_report_path
        )
        
        return result
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        
        try:
            # 尝试加载PyTorch模型
            if os.path.isfile(model_path) and model_path.endswith('.pth'):
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, nn.Module):
                    return model
            
            # 尝试加载Hugging Face模型
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                return model
            except ImportError:
                raise ImportError("需要安装transformers库来加载Hugging Face模型")
        
        except Exception as e:
            raise ValueError(f"无法加载模型 {model_path}: {e}")
    
    def _get_model_size(self, model: nn.Module) -> int:
        """获取模型大小（字节）"""
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _prepare_calibration_data(self, job: CompressionJob) -> Optional[torch.utils.data.DataLoader]:
        """准备校准数据"""
        
        # 这里应该根据任务配置加载实际的校准数据
        # 目前返回一个简化的数据加载器作为示例
        
        try:
            from torch.utils.data import DataLoader, TensorDataset
            
            # 创建示例数据
            vocab_size = 50000
            seq_length = 512
            batch_size = 8
            num_samples = job.quantization_config.calibration_dataset_size if job.quantization_config else 128
            
            # 生成随机token序列
            input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
            dataset = TensorDataset(input_ids)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            return dataloader
            
        except Exception as e:
            self.logger.warning(f"准备校准数据失败: {e}")
            return None
    
    def _prepare_training_data(self, job: CompressionJob) -> Optional[torch.utils.data.DataLoader]:
        """准备训练数据"""
        
        # 类似校准数据的简化实现
        return self._prepare_calibration_data(job)
    
    def _update_job_status(self, job_id: str, stage: str, progress: float, message: str) -> None:
        """更新任务状态"""
        
        with self.job_lock:
            if job_id in self.job_status:
                status = self.job_status[job_id]
                status.current_stage = stage
                status.progress_percent = progress
                status.last_update = utc_now()
                status.logs.append(f"[{stage}] {message}")
                
                # 限制日志数量
                if len(status.logs) > 100:
                    status.logs = status.logs[-50:]
        
        self.logger.info(f"[{job_id}] {stage}: {message} ({progress:.1f}%)")
    
    def _validate_job(self, job: CompressionJob) -> bool:
        """验证任务配置"""
        
        if not job.model_path:
            return False
        
        if job.compression_method == CompressionMethod.QUANTIZATION:
            if not job.quantization_config:
                return False
            return self.quantization_engine.validate_config(job.quantization_config)
        
        elif job.compression_method == CompressionMethod.DISTILLATION:
            if not job.distillation_config:
                return False
            trainer = DistillationTrainer(job.distillation_config)
            return trainer.validate_config(job.distillation_config)
        
        elif job.compression_method == CompressionMethod.PRUNING:
            if not job.pruning_config:
                return False
            engine = PruningEngine(job.pruning_config)
            return engine.validate_config(job.pruning_config)
        
        elif job.compression_method == CompressionMethod.MIXED:
            # 至少需要一种压缩配置
            return (job.quantization_config is not None or 
                   job.distillation_config is not None or 
                   job.pruning_config is not None)
        
        return False
    
    def _save_job_config(self, job: CompressionJob, job_dir: Path) -> None:
        """保存任务配置"""
        
        config_file = job_dir / "job_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(job.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _save_job_result(self, job_id: str, result: CompressionResult) -> None:
        """保存任务结果"""
        
        job_dir = self.workspace_dir / job_id
        result_file = job_dir / "result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    def cleanup_job(self, job_id: str, keep_result: bool = True) -> bool:
        """清理任务工作目录"""
        
        job_dir = self.workspace_dir / job_id
        
        if not job_dir.exists():
            return False
        
        try:
            if keep_result:
                # 只保留结果文件
                for file_path in job_dir.iterdir():
                    if file_path.name not in ["result.json", "evaluation_report.json"]:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
            else:
                # 删除整个目录
                shutil.rmtree(job_dir)
            
            # 从状态缓存中移除
            with self.job_lock:
                self.job_status.pop(job_id, None)
            
            self.logger.info(f"清理任务目录: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"清理任务目录失败 {job_id}: {e}")
            return False
    
    def get_active_jobs(self) -> List[str]:
        """获取活跃任务列表"""
        
        with self.job_lock:
            return list(self.active_jobs.keys())
    
    def get_job_queue_length(self) -> int:
        """获取任务队列长度"""
        
        with self.job_lock:
            return len(self.job_queue)
    
    def shutdown(self) -> None:
        """关闭流水线"""
        
        self.logger.info("关闭压缩流水线")
        
        # 等待所有任务完成
        self.executor.shutdown(wait=True)
        
        self.logger.info("压缩流水线已关闭")
