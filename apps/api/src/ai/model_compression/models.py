"""
模型压缩和量化工具的核心数据模型

定义压缩相关的数据结构和配置类
"""

from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import torch
import logging

logger = logging.getLogger(__name__)


class CompressionMethod(str, Enum):
    """压缩方法类型"""
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    MIXED = "mixed"


class QuantizationMethod(str, Enum):
    """量化方法"""
    PTQ = "post_training_quantization"
    QAT = "quantization_aware_training"
    GPTQ = "gptq"
    AWQ = "awq"
    SMOOTHQUANT = "smoothquant"


class PrecisionType(str, Enum):
    """精度类型"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class PruningType(str, Enum):
    """剪枝类型"""
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    MAGNITUDE = "magnitude_based"
    GRADIENT = "gradient_based"


class InferenceEngine(str, Enum):
    """推理引擎类型"""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TVM = "tvm"
    TORCHSCRIPT = "torchscript"


@dataclass
class QuantizationConfig:
    """量化配置"""
    method: QuantizationMethod
    precision: PrecisionType
    calibration_dataset_size: int = 512
    use_dynamic_quant: bool = False
    preserve_accuracy: bool = True
    target_accuracy_loss: float = 0.05
    group_size: int = 128
    desc_act: bool = False
    damp_percent: float = 0.1
    cache_calibration_data: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "method": self.method.value,
            "precision": self.precision.value,
            "calibration_dataset_size": self.calibration_dataset_size,
            "use_dynamic_quant": self.use_dynamic_quant,
            "preserve_accuracy": self.preserve_accuracy,
            "target_accuracy_loss": self.target_accuracy_loss,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "damp_percent": self.damp_percent,
            "cache_calibration_data": self.cache_calibration_data,
        }


@dataclass
class DistillationConfig:
    """蒸馏配置"""
    teacher_model: str
    student_model: str
    distillation_type: str = "response_based"  # response_based, feature_based, attention_based
    temperature: float = 3.0
    alpha: float = 0.5  # 蒸馏损失权重
    feature_layers: Optional[List[int]] = None
    num_epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 8
    warmup_steps: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "distillation_type": self.distillation_type,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "feature_layers": self.feature_layers,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
        }


@dataclass
class PruningConfig:
    """剪枝配置"""
    pruning_type: PruningType
    sparsity_ratio: float = 0.5
    structured_n: int = 2  # N:M结构化剪枝的N
    structured_m: int = 4  # N:M结构化剪枝的M
    importance_metric: str = "magnitude"  # magnitude, gradient, fisher
    gradual_pruning: bool = True
    recovery_epochs: int = 5
    recovery_learning_rate: float = 1e-5
    target_modules: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pruning_type": self.pruning_type.value,
            "sparsity_ratio": self.sparsity_ratio,
            "structured_n": self.structured_n,
            "structured_m": self.structured_m,
            "importance_metric": self.importance_metric,
            "gradual_pruning": self.gradual_pruning,
            "recovery_epochs": self.recovery_epochs,
            "recovery_learning_rate": self.recovery_learning_rate,
            "target_modules": self.target_modules,
        }


@dataclass
class CompressionJob:
    """压缩任务"""
    job_id: str = field(default_factory=lambda: str(uuid4()))
    job_name: str = "compression_job"
    model_path: str = ""
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    quantization_config: Optional[QuantizationConfig] = None
    distillation_config: Optional[DistillationConfig] = None
    pruning_config: Optional[PruningConfig] = None
    
    # 任务状态
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    created_at: datetime = field(default_factory=utc_factory)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # 输出配置
    output_dir: str = "compressed_models"
    save_intermediate: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "model_path": self.model_path,
            "compression_method": self.compression_method.value,
            "quantization_config": self.quantization_config.to_dict() if self.quantization_config else None,
            "distillation_config": self.distillation_config.to_dict() if self.distillation_config else None,
            "pruning_config": self.pruning_config.to_dict() if self.pruning_config else None,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "output_dir": self.output_dir,
            "save_intermediate": self.save_intermediate,
        }


@dataclass
class CompressionResult:
    """压缩结果"""
    job_id: str
    original_model_size: int  # bytes
    compressed_model_size: int  # bytes
    compression_ratio: float
    original_params: int
    compressed_params: int
    param_reduction_ratio: float
    
    # 性能指标
    original_latency: float = 0.0  # ms
    compressed_latency: float = 0.0  # ms
    latency_improvement: float = 0.0
    original_memory: int = 0  # MB
    compressed_memory: int = 0  # MB
    memory_reduction: float = 0.0
    
    # 精度指标
    original_accuracy: float = 0.0
    compressed_accuracy: float = 0.0
    accuracy_loss: float = 0.0
    
    # 输出路径
    compressed_model_path: str = ""
    evaluation_report_path: str = ""
    
    # 压缩信息
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    compression_config: Dict[str, Any] = field(default_factory=dict)
    compression_time: float = 0.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "job_id": self.job_id,
            "original_model_size": self.original_model_size,
            "compressed_model_size": self.compressed_model_size,
            "compression_ratio": self.compression_ratio,
            "original_params": self.original_params,
            "compressed_params": self.compressed_params,
            "param_reduction_ratio": self.param_reduction_ratio,
            "original_latency": self.original_latency,
            "compressed_latency": self.compressed_latency,
            "latency_improvement": self.latency_improvement,
            "original_memory": self.original_memory,
            "compressed_memory": self.compressed_memory,
            "memory_reduction": self.memory_reduction,
            "original_accuracy": self.original_accuracy,
            "compressed_accuracy": self.compressed_accuracy,
            "accuracy_loss": self.accuracy_loss,
            "compressed_model_path": self.compressed_model_path,
            "evaluation_report_path": self.evaluation_report_path,
            "compression_method": self.compression_method.value,
            "compression_config": self.compression_config,
            "compression_time": self.compression_time,
        }


class HardwareBenchmark(TypedDict):
    """硬件基准测试"""
    device_name: str
    device_type: str  # cpu, gpu, tpu, npu
    batch_size: int
    sequence_length: int
    throughput: float  # tokens/second
    latency_p50: float  # ms
    latency_p95: float  # ms
    latency_p99: float  # ms
    memory_usage: float  # MB
    power_consumption: Optional[float]  # watts


@dataclass
class CompressionStrategy:
    """压缩策略"""
    strategy_name: str
    description: str
    target_scenario: str  # cloud, edge, mobile
    compression_methods: List[CompressionMethod]
    expected_compression_ratio: float
    expected_speedup: float
    expected_accuracy_retention: float
    hardware_compatibility: List[str]
    config_template: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "description": self.description,
            "target_scenario": self.target_scenario,
            "compression_methods": [method.value for method in self.compression_methods],
            "expected_compression_ratio": self.expected_compression_ratio,
            "expected_speedup": self.expected_speedup,
            "expected_accuracy_retention": self.expected_accuracy_retention,
            "hardware_compatibility": self.hardware_compatibility,
            "config_template": self.config_template,
        }


@dataclass
class ModelInfo:
    """模型信息"""
    model_name: str
    model_type: str  # transformer, cnn, rnn
    architecture: str  # llama, gpt, bert
    num_parameters: int
    model_size: int  # bytes
    precision: PrecisionType
    supported_frameworks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "num_parameters": self.num_parameters,
            "model_size": self.model_size,
            "precision": self.precision.value,
            "supported_frameworks": self.supported_frameworks,
        }


# 预定义的压缩策略模板
DEFAULT_COMPRESSION_STRATEGIES = [
    CompressionStrategy(
        strategy_name="Mobile Optimization",
        description="面向移动设备的极致压缩策略",
        target_scenario="mobile",
        compression_methods=[CompressionMethod.QUANTIZATION, CompressionMethod.PRUNING],
        expected_compression_ratio=8.0,
        expected_speedup=4.0,
        expected_accuracy_retention=0.9,
        hardware_compatibility=["arm", "mobile_gpu"],
        config_template={
            "quantization": {"precision": "int4", "calibration_dataset_size": 256},
            "pruning": {"sparsity_ratio": 0.7, "pruning_type": "structured"}
        }
    ),
    CompressionStrategy(
        strategy_name="Edge Computing",
        description="边缘计算设备优化策略",
        target_scenario="edge",
        compression_methods=[CompressionMethod.QUANTIZATION],
        expected_compression_ratio=4.0,
        expected_speedup=2.0,
        expected_accuracy_retention=0.95,
        hardware_compatibility=["arm", "x86", "edge_tpu"],
        config_template={
            "quantization": {"precision": "int8", "calibration_dataset_size": 512}
        }
    ),
    CompressionStrategy(
        strategy_name="Cloud Inference",
        description="云端推理服务优化策略",
        target_scenario="cloud",
        compression_methods=[CompressionMethod.QUANTIZATION, CompressionMethod.DISTILLATION],
        expected_compression_ratio=2.0,
        expected_speedup=1.5,
        expected_accuracy_retention=0.98,
        hardware_compatibility=["cuda", "tensorrt", "a100", "v100"],
        config_template={
            "quantization": {"precision": "fp16", "calibration_dataset_size": 1024},
            "distillation": {"temperature": 4.0, "alpha": 0.7}
        }
    )
]