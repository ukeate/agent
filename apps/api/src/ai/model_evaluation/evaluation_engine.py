import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Any, Callable, Union
import json
import asyncio
import time
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
import warnings
import evaluate
from lm_eval import evaluator
from lm_eval.models import huggingface
from enum import Enum

from src.core.logging import get_logger
logger = get_logger(__name__)

warnings.filterwarnings("ignore")

class EvaluationStatus(str, Enum):
    """评估状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class EvaluationConfig:
    model_name: str = ""
    model_path: str = ""
    task_type: str = "text_generation"  # 'text_generation', 'classification', 'qa', 'summarization'
    device: str = "auto"
    batch_size: int = 4
    max_length: int = 512
    max_seq_length: int = 512
    num_samples: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    precision: str = "fp16"  # fp16, fp32, bf16
    enable_optimizations: bool = True
    # 缓存相关配置
    enable_caching: bool = True
    cache_dir: str = "cache/evaluations"
    cache_expiry_hours: int = 24
    # 并发控制
    max_concurrent_evaluations: int = 2
    # 错误处理
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 3600

@dataclass
class BenchmarkConfig:
    name: str
    tasks: List[str]
    num_fewshot: int = 0
    limit: Optional[int] = None
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: Optional[str] = None
    log_samples: bool = True

@dataclass
class EvaluationMetrics:
    accuracy: float
    f1_score: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    perplexity: Optional[float] = None
    inference_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    custom_metrics: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResult:
    model_name: str
    benchmark_name: str
    task_name: str
    metrics: EvaluationMetrics
    config: EvaluationConfig
    timestamp: datetime
    duration: float
    error: Optional[str] = None
    samples_evaluated: int = 0
    hardware_info: Optional[Dict[str, Any]] = None

class ModelEvaluationEngine:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.metrics_cache = {}
        
        # 任务管理
        self.current_jobs = {}
        self.job_history = []
        self.is_running = False
        
        self._initialize_model()
        
    def load_model(self) -> None:
        """加载模型和tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            start_time = time.time()
            
            # 设置模型配置
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": self._get_torch_dtype(),
                "device_map": "auto" if self.config.device == "cuda" else None,
            }
            
            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                
            # 加载配置
            config = AutoConfig.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                config=config,
                **model_kwargs
            )
            
            if self.config.enable_optimizations:
                self._apply_optimizations()
                
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_torch_dtype(self):
        """获取torch数据类型"""
        if self.config.precision == "fp16":
            return torch.float16
        elif self.config.precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def _apply_optimizations(self):
        """应用模型优化"""
        try:
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"Failed to apply optimizations: {e}")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_name": torch.cuda.get_device_name()
            })
            
        return info
    
    def evaluate_with_lm_eval(self, benchmark_config: BenchmarkConfig) -> List[EvaluationResult]:
        """使用lm-evaluation-harness进行评估"""
        if not self.model:
            self.load_model()
            
        results = []
        start_time = time.time()
        
        try:
            # 创建HuggingFace模型实例
            hf_model = huggingface.HFLM(
                pretrained=self.config.model_path,
                device=self.config.device,
                batch_size=benchmark_config.batch_size,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # 运行评估
            eval_results = evaluator.simple_evaluate(
                model=hf_model,
                tasks=benchmark_config.tasks,
                num_fewshot=benchmark_config.num_fewshot,
                limit=benchmark_config.limit,
                batch_size=benchmark_config.batch_size,
                device=benchmark_config.device,
                log_samples=benchmark_config.log_samples
            )
            
            # 处理结果
            for task_name, task_results in eval_results["results"].items():
                metrics = EvaluationMetrics(
                    accuracy=task_results.get("acc", 0.0),
                    f1_score=task_results.get("f1", None),
                    bleu_score=task_results.get("bleu", None),
                    rouge_scores=self._extract_rouge_scores(task_results),
                    perplexity=task_results.get("ppl", None),
                    custom_metrics=task_results
                )
                
                result = EvaluationResult(
                    model_name=self.config.model_name,
                    benchmark_name=benchmark_config.name,
                    task_name=task_name,
                    metrics=metrics,
                    config=self.config,
                    timestamp=utc_now(),
                    duration=time.time() - start_time,
                    samples_evaluated=task_results.get("num_samples", 0),
                    hardware_info=self._get_hardware_info()
                )
                
                results.append(result)
                
        except Exception as e:
            error_result = EvaluationResult(
                model_name=self.config.model_name,
                benchmark_name=benchmark_config.name,
                task_name="failed",
                metrics=EvaluationMetrics(accuracy=0.0),
                config=self.config,
                timestamp=utc_now(),
                duration=time.time() - start_time,
                error=str(e),
                hardware_info=self._get_hardware_info()
            )
            results.append(error_result)
            logger.error(f"Evaluation failed: {e}")
            
        return results
    
    def _extract_rouge_scores(self, task_results: Dict) -> Optional[Dict[str, float]]:
        """提取ROUGE评分"""
        rouge_scores = {}
        for key, value in task_results.items():
            if key.startswith("rouge"):
                rouge_scores[key] = value
        return rouge_scores if rouge_scores else None
    
    def evaluate_custom_metrics(self, dataset, metrics: List[str]) -> EvaluationMetrics:
        """评估自定义指标"""
        if not self.model:
            self.load_model()
            
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        results = {}
        total_samples = 0
        
        try:
            for metric_name in metrics:
                if metric_name == "accuracy":
                    results["accuracy"] = self._compute_accuracy(dataset)
                elif metric_name == "perplexity":
                    results["perplexity"] = self._compute_perplexity(dataset)
                elif metric_name == "bleu":
                    results["bleu_score"] = self._compute_bleu(dataset)
                elif metric_name == "rouge":
                    results["rouge_scores"] = self._compute_rouge(dataset)
                    
            inference_time = time.time() - start_time
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_usage = (memory_after - memory_before) / 1024 / 1024  # MB
            
            return EvaluationMetrics(
                accuracy=results.get("accuracy", 0.0),
                f1_score=results.get("f1_score"),
                bleu_score=results.get("bleu_score"),
                rouge_scores=results.get("rouge_scores"),
                perplexity=results.get("perplexity"),
                inference_time=inference_time,
                memory_usage=memory_usage,
                throughput=total_samples / inference_time if inference_time > 0 else 0.0,
                custom_metrics=results
            )
            
        except Exception as e:
            logger.error(f"Custom evaluation failed: {e}")
            return EvaluationMetrics(accuracy=0.0, custom_metrics={"error": str(e)})
    
    def _compute_accuracy(self, dataset) -> float:
        """计算准确率"""
        accuracy_metric = evaluate.load("accuracy")
        correct = 0
        total = 0
        
        for batch in dataset:
            predictions = self._generate_predictions(batch["input"])
            labels = batch["target"]
            
            batch_correct = sum(p == l for p, l in zip(predictions, labels))
            correct += batch_correct
            total += len(predictions)
            
        return correct / total if total > 0 else 0.0
    
    def _compute_perplexity(self, dataset) -> float:
        """计算困惑度"""
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
                
        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def _compute_bleu(self, dataset) -> float:
        """计算BLEU分数"""
        bleu_metric = evaluate.load("bleu")
        predictions = []
        references = []
        
        for batch in dataset:
            batch_predictions = self._generate_predictions(batch["input"])
            predictions.extend(batch_predictions)
            references.extend(batch["target"])
            
        return bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    
    def _compute_rouge(self, dataset) -> Dict[str, float]:
        """计算ROUGE分数"""
        rouge_metric = evaluate.load("rouge")
        predictions = []
        references = []
        
        for batch in dataset:
            batch_predictions = self._generate_predictions(batch["input"])
            predictions.extend(batch_predictions)
            references.extend(batch["target"])
            
        return rouge_metric.compute(predictions=predictions, references=references)
    
    def _generate_predictions(self, inputs: List[str]) -> List[str]:
        """生成预测结果"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for input_text in inputs:
                encoded = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **encoded,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                predictions.append(generated_text)
                
        return predictions
    
    async def evaluate_async(self, benchmark_configs: List[BenchmarkConfig]) -> List[EvaluationResult]:
        """异步评估多个基准测试"""
        tasks = []
        for config in benchmark_configs:
            task = asyncio.create_task(
                self._evaluate_single_async(config)
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = EvaluationResult(
                    model_name=self.config.model_name,
                    benchmark_name=benchmark_configs[i].name,
                    task_name="failed",
                    metrics=EvaluationMetrics(accuracy=0.0),
                    config=self.config,
                    timestamp=utc_now(),
                    duration=0.0,
                    error=str(result),
                    hardware_info=self._get_hardware_info()
                )
                processed_results.append(error_result)
            else:
                processed_results.extend(result)
                
        return processed_results
    
    async def _evaluate_single_async(self, benchmark_config: BenchmarkConfig) -> List[EvaluationResult]:
        """异步评估单个基准测试"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, 
                self.evaluate_with_lm_eval, 
                benchmark_config
            )
        return results
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info(f"初始化模型: {self.config.model_path}")
            
            # 检查是否为本地路径
            if os.path.exists(self.config.model_path):
                model_path = self.config.model_path
            else:
                model_path = self.config.model_path
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, self.config.dtype),
                device_map=self.device if self.config.device != "auto" else "auto",
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                low_cpu_mem_usage=True
            )
            
            # 如果指定了具体设备，移动模型
            if self.config.device not in ["auto", "cuda"]:
                self.model = self.model.to(self.device)
                
            logger.info(f"模型加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise RuntimeError(f"模型初始化失败: {e}")
    
    async def start_evaluation(self, benchmark_name: str, model_name: str = None) -> str:
        """启动模型评估任务"""
        if model_name is None:
            model_name = self.config.model_path.split('/')[-1]
        
        job_id = self._generate_job_id(model_name, benchmark_name)
        
        # 检查是否已有同名任务在运行
        for existing_job_id, job_info in self.current_jobs.items():
            if (job_info.get('model_name') == model_name and 
                job_info.get('benchmark_name') == benchmark_name and
                job_info.get('status') == EvaluationStatus.RUNNING):
                logger.warning(f"任务已在运行: {existing_job_id}")
                return existing_job_id
        
        # 创建任务记录
        self.current_jobs[job_id] = {
            'status': EvaluationStatus.PENDING,
            'model_name': model_name,
            'benchmark_name': benchmark_name,
            'created_at': utc_now(),
            'task': None
        }
        
        # 启动异步任务
        task = asyncio.create_task(self._run_evaluation_task(job_id, benchmark_name))
        self.current_jobs[job_id]['task'] = task
        
        logger.info(f"评估任务已启动: {job_id}")
        return job_id
    
    async def _run_evaluation_task(self, job_id: str, benchmark_name: str):
        """运行评估任务的异步方法"""
        try:
            self._update_job_status(job_id, EvaluationStatus.RUNNING)
            
            # 初始化模型（如果还没初始化）
            if self.model is None:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._initialize_model
                )
            
            # 执行评估（在线程池中执行，避免阻塞事件循环）
            results = await asyncio.get_running_loop().run_in_executor(
                None, self._execute_benchmark, benchmark_name
            )
            
            # 保存结果
            self.current_jobs[job_id]['results'] = results
            self._update_job_status(job_id, EvaluationStatus.COMPLETED)
            
            logger.info(f"评估任务完成: {job_id}")
            
        except Exception as e:
            logger.error(f"评估任务失败 {job_id}: {e}")
            self.current_jobs[job_id]['error'] = str(e)
            self._update_job_status(job_id, EvaluationStatus.FAILED)
            
    def _execute_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """执行基准测试"""
        try:
            # 这里实现具体的基准测试逻辑
            from ai.model_evaluation.benchmark_manager import BenchmarkConfig
            
            # 创建基准测试配置
            benchmark_config = BenchmarkConfig(
                tasks=[benchmark_name],
                batch_size=self.config.batch_size,
                num_fewshot=0,  # 可配置
                limit=None,
                seed=42
            )
            
            # 执行评估
            results = self.evaluate_with_lm_eval(benchmark_config)
            
            return {
                'benchmark_name': benchmark_name,
                'results': results,
                'timestamp': utc_now().isoformat(),
                'model_name': self.config.model_path,
                'config': asdict(self.config)
            }
            
        except Exception as e:
            logger.error(f"基准测试执行失败: {e}")
            raise
    
    def _generate_job_id(self, model_name: str, benchmark_name: str) -> str:
        """生成唯一的任务ID"""
        timestamp = utc_now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{model_name}_{benchmark_name}_{timestamp}"
    
    def get_job_status(self, job_id: str) -> Optional[EvaluationStatus]:
        """获取任务状态"""
        if job_id not in self.current_jobs:
            return None
        return self.current_jobs[job_id].get('status')
    
    def _update_job_status(self, job_id: str, status: EvaluationStatus):
        """更新任务状态"""
        if job_id in self.current_jobs:
            self.current_jobs[job_id]['status'] = status
    
    async def stop_evaluation(self, job_id: str) -> bool:
        """停止评估任务"""
        if job_id not in self.current_jobs:
            return False
        
        job = self.current_jobs[job_id]
        if 'task' in job and not job['task'].done():
            job['task'].cancel()
        
        self._update_job_status(job_id, EvaluationStatus.CANCELLED)
        return True
    
    def _aggregate_results(self, raw_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """聚合评估结果"""
        aggregated = {}
        
        # 计算整体准确率
        accuracy_values = [task_results.get('accuracy', 0.0) for task_results in raw_results.values()]
        if accuracy_values:
            aggregated['overall_accuracy'] = sum(accuracy_values) / len(accuracy_values)
        
        # 计算整体F1分数
        f1_values = [task_results.get('f1', 0.0) for task_results in raw_results.values()]
        if f1_values:
            aggregated['overall_f1'] = sum(f1_values) / len(f1_values)
        
        # 保存任务级别结果
        aggregated['task_results'] = raw_results
        
        return aggregated
    
    def _check_resource_availability(self) -> bool:
        """检查资源可用性"""
        running_jobs = [job for job in self.current_jobs.values() 
                       if job.get('status') == EvaluationStatus.RUNNING]
        return len(running_jobs) < self.config.max_concurrent_evaluations
    
    def _cleanup_completed_jobs(self) -> int:
        """清理已完成的任务"""
        completed_statuses = [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]
        completed_jobs = [job_id for job_id, job in self.current_jobs.items()
                         if job.get('status') in completed_statuses]
        
        for job_id in completed_jobs:
            self.job_history.append(self.current_jobs[job_id])
            del self.current_jobs[job_id]
        
        return len(completed_jobs)
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleanup completed")

class BatchEvaluationManager:
    """批量评估管理器"""
    
    def __init__(self, max_concurrent_evaluations: int = 3):
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.active_evaluations = {}
        self.results_storage = []
        
    async def evaluate_multiple_models(
        self, 
        model_configs: List[EvaluationConfig],
        benchmark_configs: List[BenchmarkConfig]
    ) -> Dict[str, List[EvaluationResult]]:
        """评估多个模型"""
        
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        tasks = []
        
        for model_config in model_configs:
            task = asyncio.create_task(
                self._evaluate_model_with_semaphore(
                    semaphore, model_config, benchmark_configs
                )
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 组织结果
        organized_results = {}
        for i, result in enumerate(results):
            model_name = model_configs[i].model_name
            if isinstance(result, Exception):
                logger.error(f"Model {model_name} evaluation failed: {result}")
                organized_results[model_name] = []
            else:
                organized_results[model_name] = result
                
        return organized_results
    
    async def _evaluate_model_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        model_config: EvaluationConfig,
        benchmark_configs: List[BenchmarkConfig]
    ) -> List[EvaluationResult]:
        """使用信号量控制并发的模型评估"""
        
        async with semaphore:
            engine = ModelEvaluationEngine(model_config)
            try:
                results = await engine.evaluate_async(benchmark_configs)
                return results
            finally:
                engine.cleanup()
