"""推理服务引擎"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from queue import Queue
import threading
import json
import torch
import torch.nn as nn
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from cachetools import TTLCache, LRUCache
from .registry import ModelRegistry, ModelFormat, PyTorchLoader, ONNXLoader, HuggingFaceLoader

logger = get_logger(__name__)

class InferenceStatus(str, Enum):
    """推理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    model_name: str
    model_version: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = None
    batch_size: int = 1
    timeout_seconds: int = 30
    created_at: datetime = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    status: InferenceStatus
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, max_batch_size: int = 16, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: Queue = Queue()
        self.results: Dict[str, InferenceResult] = {}
        self.is_processing = False
        self.lock = threading.Lock()
    
    def add_request(self, request: InferenceRequest) -> str:
        """添加推理请求"""
        self.pending_requests.put(request)
        return request.request_id
    
    def get_result(self, request_id: str) -> Optional[InferenceResult]:
        """获取推理结果"""
        return self.results.get(request_id)
    
    def process_batch(self, model_instance, model_format: ModelFormat) -> List[InferenceResult]:
        """处理批量请求"""
        batch_requests = []
        
        # 收集批量请求
        start_time = time.time()
        while (len(batch_requests) < self.max_batch_size and 
               (time.time() - start_time) < self.max_wait_time):
            
            if not self.pending_requests.empty():
                try:
                    request = self.pending_requests.get_nowait()
                    batch_requests.append(request)
                except:
                    break
            else:
                time.sleep(0.001)  # 短暂等待
        
        if not batch_requests:
            return []
        
        # 执行批量推理
        results = []
        for request in batch_requests:
            start_time = time.time()
            try:
                outputs = self._execute_single_inference(model_instance, request, model_format)
                processing_time = (time.time() - start_time) * 1000
                
                result = InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.COMPLETED,
                    outputs=outputs,
                    processing_time_ms=processing_time,
                    completed_at=datetime.now(timezone.utc)
                )
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                result = InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=str(e),
                    processing_time_ms=processing_time,
                    completed_at=datetime.now(timezone.utc)
                )
            
            results.append(result)
            self.results[request.request_id] = result
        
        return results
    
    def _execute_single_inference(self, model_instance, request: InferenceRequest, model_format: ModelFormat):
        """执行单个推理"""
        if model_format == ModelFormat.PYTORCH:
            return self._pytorch_inference(model_instance, request)
        elif model_format == ModelFormat.ONNX:
            return self._onnx_inference(model_instance, request)
        elif model_format == ModelFormat.HUGGINGFACE:
            return self._huggingface_inference(model_instance, request)
        else:
            raise ValueError(f"不支持的模型格式: {model_format}")
    
    def _pytorch_inference(self, model: nn.Module, request: InferenceRequest) -> Dict[str, Any]:
        """PyTorch模型推理"""
        model.eval()
        
        with torch.no_grad():
            # 处理输入
            inputs = {}
            for key, value in request.inputs.items():
                if isinstance(value, list):
                    inputs[key] = torch.tensor(value)
                elif isinstance(value, np.ndarray):
                    inputs[key] = torch.from_numpy(value)
                else:
                    inputs[key] = value
            
            # 执行推理
            if len(inputs) == 1:
                # 单输入模型
                input_tensor = list(inputs.values())[0]
                outputs = model(input_tensor)
            else:
                # 多输入模型
                outputs = model(**inputs)
            
            # 处理输出
            if isinstance(outputs, torch.Tensor):
                return {"output": outputs.cpu().numpy().tolist()}
            elif isinstance(outputs, tuple):
                return {f"output_{i}": out.cpu().numpy().tolist() if isinstance(out, torch.Tensor) else out 
                       for i, out in enumerate(outputs)}
            elif isinstance(outputs, dict):
                return {key: val.cpu().numpy().tolist() if isinstance(val, torch.Tensor) else val 
                       for key, val in outputs.items()}
            else:
                return {"output": str(outputs)}
    
    def _onnx_inference(self, session: ort.InferenceSession, request: InferenceRequest) -> Dict[str, Any]:
        """ONNX模型推理"""
        # 准备输入
        ort_inputs = {}
        for key, value in request.inputs.items():
            if isinstance(value, list):
                ort_inputs[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, np.ndarray):
                ort_inputs[key] = value.astype(np.float32)
            else:
                ort_inputs[key] = np.array(value, dtype=np.float32)
        
        # 执行推理
        outputs = session.run(None, ort_inputs)
        
        # 处理输出
        output_names = [output.name for output in session.get_outputs()]
        return {name: output.tolist() for name, output in zip(output_names, outputs)}
    
    def _huggingface_inference(self, model_data: tuple, request: InferenceRequest) -> Dict[str, Any]:
        """HuggingFace模型推理"""
        model, tokenizer, config = model_data
        
        # 处理文本输入
        if "text" in request.inputs:
            text = request.inputs["text"]
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model(**encoded)
            
            # 处理输出
            if hasattr(outputs, 'last_hidden_state'):
                result = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
            elif hasattr(outputs, 'logits'):
                result = outputs.logits.cpu().numpy().tolist()
            else:
                result = str(outputs)
            
            return {"embeddings": result} if hasattr(outputs, 'last_hidden_state') else {"logits": result}
        else:
            raise ValueError("HuggingFace模型需要文本输入")

class ModelInstance:
    """模型实例"""
    
    def __init__(self, metadata, model_path: str, device: str = "cpu"):
        self.metadata = metadata
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_time = None
        self.last_used = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        start_time = time.time()
        
        try:
            if self.metadata.format == ModelFormat.PYTORCH:
                self.model = PyTorchLoader.load_model(self.model_path, self.device)
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
            elif self.metadata.format == ModelFormat.ONNX:
                providers = ['CPUExecutionProvider']
                if self.device == 'cuda' and ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.model = ort.InferenceSession(self.model_path, providers=providers)
            elif self.metadata.format == ModelFormat.HUGGINGFACE:
                self.model = HuggingFaceLoader.load_model_and_tokenizer(self.model_path)
            
            self.load_time = time.time() - start_time
            self.last_used = datetime.now(timezone.utc)
            logger.info(f"模型加载成功: {self.metadata.name}:{self.metadata.version}, 耗时: {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """检查模型是否健康"""
        return self.model is not None
    
    def update_last_used(self):
        """更新最后使用时间"""
        self.last_used = datetime.now(timezone.utc)

class InferenceEngine:
    """推理服务引擎"""
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        max_concurrent_requests: int = 100,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.max_concurrent_requests = max_concurrent_requests
        self.device = device
        
        # 模型缓存
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.model_cache = LRUCache(maxsize=10)  # 最多缓存10个模型
        
        # 结果缓存
        self.result_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        
        # 批处理器
        self.batch_processors: Dict[str, BatchProcessor] = {}
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # 性能指标
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0
        }
        self.metrics_lock = threading.Lock()
    
    def _get_model_key(self, name: str, version: str) -> str:
        """生成模型缓存键"""
        return f"{name}:{version}"
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """生成结果缓存键"""
        inputs_str = json.dumps(request.inputs, sort_keys=True)
        params_str = json.dumps(request.parameters, sort_keys=True)
        return f"{request.model_name}:{request.model_version}:{hash(inputs_str + params_str)}"
    
    def load_model(self, name: str, version: str = "latest") -> bool:
        """加载模型到内存"""
        model_key = self._get_model_key(name, version)
        
        if model_key in self.loaded_models:
            logger.info(f"模型已加载: {model_key}")
            return True
        
        try:
            # 获取模型元数据
            metadata = self.model_registry.get_model(name, version)
            if not metadata:
                logger.error(f"模型不存在: {name}:{version}")
                return False
            
            # 获取模型文件路径
            model_path = self.model_registry.get_model_path(name, metadata.version)
            if not model_path:
                logger.error(f"模型文件不存在: {name}:{metadata.version}")
                return False
            
            # 创建模型实例
            model_instance = ModelInstance(metadata, model_path, self.device)
            
            # 缓存模型
            self.loaded_models[model_key] = model_instance
            self.model_cache[model_key] = model_instance
            
            # 创建批处理器
            self.batch_processors[model_key] = BatchProcessor()
            
            logger.info(f"模型加载成功: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def unload_model(self, name: str, version: str = "latest") -> bool:
        """卸载模型"""
        model_key = self._get_model_key(name, version)
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            
            if model_key in self.model_cache:
                del self.model_cache[model_key]
            
            if model_key in self.batch_processors:
                del self.batch_processors[model_key]
            
            # 清理GPU内存
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            logger.info(f"模型卸载成功: {model_key}")
            return True
        
        return False
    
    async def inference(self, request: InferenceRequest) -> InferenceResult:
        """执行推理"""
        # 更新指标
        with self.metrics_lock:
            self.metrics["total_requests"] += 1
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(request)
            if cache_key in self.result_cache:
                with self.metrics_lock:
                    self.metrics["cache_hits"] += 1
                    self.metrics["successful_requests"] += 1
                
                cached_result = self.result_cache[cache_key]
                # 创建新的结果对象，但保持原始的处理时间
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.COMPLETED,
                    outputs=cached_result.outputs,
                    processing_time_ms=cached_result.processing_time_ms,
                    completed_at=datetime.now(timezone.utc)
                )
            
            with self.metrics_lock:
                self.metrics["cache_misses"] += 1
            
            # 确保模型已加载
            model_key = self._get_model_key(request.model_name, request.model_version)
            if not await self._ensure_model_loaded(request.model_name, request.model_version):
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=f"无法加载模型: {request.model_name}:{request.model_version}"
                )
            
            # 获取模型实例和批处理器
            model_instance = self.loaded_models[model_key]
            batch_processor = self.batch_processors[model_key]
            
            # 添加到批处理队列
            batch_processor.add_request(request)
            
            # 在线程池中执行批处理
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.executor, 
                batch_processor.process_batch,
                model_instance.model,
                model_instance.metadata.format
            )
            
            # 获取结果
            result = batch_processor.get_result(request.request_id)
            if not result:
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error="推理结果未找到"
                )
            
            # 更新模型使用时间
            model_instance.update_last_used()
            
            # 缓存成功的结果
            if result.status == InferenceStatus.COMPLETED:
                self.result_cache[cache_key] = result
                with self.metrics_lock:
                    self.metrics["successful_requests"] += 1
                    # 更新平均处理时间
                    self.metrics["average_processing_time"] = (
                        (self.metrics["average_processing_time"] * (self.metrics["successful_requests"] - 1) + 
                         result.processing_time_ms) / self.metrics["successful_requests"]
                    )
            else:
                with self.metrics_lock:
                    self.metrics["failed_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            with self.metrics_lock:
                self.metrics["failed_requests"] += 1
            
            return InferenceResult(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=str(e)
            )
    
    async def _ensure_model_loaded(self, name: str, version: str) -> bool:
        """确保模型已加载"""
        model_key = self._get_model_key(name, version)
        
        if model_key not in self.loaded_models:
            return self.load_model(name, version)
        
        # 检查模型健康状态
        if not self.loaded_models[model_key].is_healthy():
            # 重新加载模型
            self.unload_model(name, version)
            return self.load_model(name, version)
        
        return True
    
    async def batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """批量推理"""
        tasks = [self.inference(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """获取已加载的模型列表"""
        result = []
        for model_key, instance in self.loaded_models.items():
            result.append({
                "model_key": model_key,
                "name": instance.metadata.name,
                "version": instance.metadata.version,
                "format": instance.metadata.format.value,
                "load_time": instance.load_time,
                "last_used": instance.last_used.isoformat() if instance.last_used else None,
                "is_healthy": instance.is_healthy()
            })
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def clear_cache(self):
        """清空缓存"""
        self.result_cache.clear()
        logger.info("推理结果缓存已清空")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        loaded_models = len(self.loaded_models)
        healthy_models = sum(1 for instance in self.loaded_models.values() if instance.is_healthy())
        
        return {
            "status": "healthy" if loaded_models == healthy_models else "degraded",
            "loaded_models": loaded_models,
            "healthy_models": healthy_models,
            "cache_size": len(self.result_cache),
            "metrics": self.get_metrics()
        }
from src.core.logging import get_logger
