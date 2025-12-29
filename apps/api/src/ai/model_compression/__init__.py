"""
模型压缩和量化工具模块

提供完整的模型压缩功能:
- 量化引擎：支持PTQ、QAT量化算法
- 知识蒸馏：实现Teacher-Student蒸馏和多种蒸馏策略
- 模型剪枝：支持结构化和非结构化剪枝
- 压缩评估：全面的性能评估和对比分析
- 压缩流水线：自动化的压缩任务管理
"""

from .models import (
    # 枚举类型
    CompressionMethod,
    QuantizationMethod,
    PrecisionType,
    PruningType,
    InferenceEngine,
    
    # 配置类
    QuantizationConfig,
    DistillationConfig,
    PruningConfig,
    
    # 数据类
    CompressionJob,
    CompressionResult,
    CompressionStrategy,
    ModelInfo,
    HardwareBenchmark,
    
    # 预定义策略
    DEFAULT_COMPRESSION_STRATEGIES,
)
from .quantization_engine import QuantizationEngine
from .distillation_trainer import (
    DistillationTrainer,
    DistillationResult,
)
from .pruning_engine import (
    PruningEngine,
    PruningResult,
)
from .compression_evaluator import (
    CompressionEvaluator,
    EvaluationMetrics,
    ComparisonReport,
)
from .compression_pipeline import (
    CompressionPipeline,
    PipelineStatus,

# 版本信息
)

__version__ = "1.0.0"

# 导出的主要类和函数
__all__ = [
    # 枚举和常量
    "CompressionMethod",
    "QuantizationMethod",
    "PrecisionType",
    "PruningType",
    "InferenceEngine",
    "DEFAULT_COMPRESSION_STRATEGIES",
    
    # 配置类
    "QuantizationConfig",
    "DistillationConfig",
    "PruningConfig",
    
    # 任务和结果类
    "CompressionJob",
    "CompressionResult",
    "CompressionStrategy",
    "ModelInfo",
    "HardwareBenchmark",
    
    # 核心引擎
    "QuantizationEngine",
    "DistillationTrainer",
    "PruningEngine",
    "CompressionEvaluator",
    "CompressionPipeline",
    
    # 结果类
    "DistillationResult",
    "PruningResult",
    "EvaluationMetrics",
    "ComparisonReport",
    "PipelineStatus",
    
    # 便捷函数
    "create_compression_job",
    "get_supported_methods",
    "get_default_config",
]

def create_compression_job(
    job_name: str,
    model_path: str,
    compression_method: CompressionMethod,
    **kwargs
) -> CompressionJob:
    """便捷函数：创建压缩任务
    
    Args:
        job_name: 任务名称
        model_path: 模型路径
        compression_method: 压缩方法
        **kwargs: 其他配置参数
        
    Returns:
        CompressionJob: 压缩任务对象
    """
    
    job = CompressionJob(
        job_name=job_name,
        model_path=model_path,
        compression_method=compression_method
    )
    
    # 根据压缩方法设置默认配置
    if compression_method == CompressionMethod.QUANTIZATION:
        job.quantization_config = kwargs.get('quantization_config') or QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
    
    elif compression_method == CompressionMethod.DISTILLATION:
        teacher_model = kwargs.get('teacher_model')
        student_model = kwargs.get('student_model')
        if not teacher_model or not student_model:
            raise ValueError("蒸馏需要指定teacher_model和student_model")
        
        job.distillation_config = kwargs.get('distillation_config') or DistillationConfig(
            teacher_model=teacher_model,
            student_model=student_model
        )
    
    elif compression_method == CompressionMethod.PRUNING:
        job.pruning_config = kwargs.get('pruning_config') or PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5
        )
    
    # 设置其他参数
    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)
    
    return job

def get_supported_methods() -> dict:
    """获取支持的压缩方法
    
    Returns:
        dict: 支持的方法和描述
    """
    
    return {
        "quantization": {
            "description": "模型量化压缩",
            "methods": [QuantizationMethod.PTQ.value, QuantizationMethod.QAT.value],
            "precisions": [precision.value for precision in PrecisionType],
        },
        "distillation": {
            "description": "知识蒸馏压缩",
            "strategies": ["response_based", "feature_based", "attention_based", "self_distillation"],
        },
        "pruning": {
            "description": "模型剪枝压缩",
            "types": [ptype.value for ptype in PruningType],
        },
        "mixed": {
            "description": "混合压缩策略",
            "combinations": ["quantization+pruning", "distillation+quantization", "all_methods"],
        }
    }

def get_default_config(compression_method: CompressionMethod) -> dict:
    """获取默认配置
    
    Args:
        compression_method: 压缩方法
        
    Returns:
        dict: 默认配置
    """
    
    if compression_method == CompressionMethod.QUANTIZATION:
        return QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8,
            calibration_dataset_size=512,
            preserve_accuracy=True,
            target_accuracy_loss=0.05
        ).to_dict()
    
    elif compression_method == CompressionMethod.DISTILLATION:
        return {
            "distillation_type": "response_based",
            "temperature": 3.0,
            "alpha": 0.5,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "batch_size": 8
        }
    
    elif compression_method == CompressionMethod.PRUNING:
        return PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5,
            importance_metric="magnitude",
            gradual_pruning=True,
            recovery_epochs=5
        ).to_dict()
    
    else:
        return {}

# 模块级别的便捷实例
_quantization_engine = None
_compression_evaluator = None
_compression_pipeline = None

def get_quantization_engine() -> QuantizationEngine:
    """获取全局量化引擎实例"""
    global _quantization_engine
    if _quantization_engine is None:
        _quantization_engine = QuantizationEngine()
    return _quantization_engine

def get_compression_evaluator() -> CompressionEvaluator:
    """获取全局压缩评估器实例"""
    global _compression_evaluator
    if _compression_evaluator is None:
        _compression_evaluator = CompressionEvaluator()
    return _compression_evaluator

def get_compression_pipeline() -> CompressionPipeline:
    """获取全局压缩流水线实例"""
    global _compression_pipeline
    if _compression_pipeline is None:
        _compression_pipeline = CompressionPipeline()
    return _compression_pipeline

# 模块信息
MODEL_COMPRESSION_INFO = {
    "name": "模型压缩和量化工具",
    "version": __version__,
    "description": "提供完整的深度学习模型压缩解决方案",
    "features": [
        "多种量化算法支持（PTQ、QAT、GPTQ、AWQ等）",
        "知识蒸馏框架（多种蒸馏策略）",
        "结构化和非结构化剪枝",
        "全面的压缩效果评估",
        "自动化压缩流水线",
        "硬件性能基准测试",
        "压缩策略推荐系统"
    ],
    "supported_models": [
        "Transformer模型（GPT、BERT、T5等）",
        "卷积神经网络（ResNet、EfficientNet等）",
        "循环神经网络（LSTM、GRU等）",
        "自定义PyTorch模型"
    ],
    "supported_frameworks": [
        "PyTorch",
        "Hugging Face Transformers",
        "ONNX（导出支持）",
        "TensorRT（优化支持）"
    ]
}
