"""
微调相关数据模型定义
"""

from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from enum import Enum
from dataclasses import dataclass
import uuid

class ModelArchitecture(str, Enum):
    """支持的模型架构"""
    LLAMA = "llama"
    MISTRAL = "mistral" 
    QWEN = "qwen"
    CHATGLM = "chatglm"
    BAICHUAN = "baichuan"
    YI = "yi"
    DEEPSEEK = "deepseek"
    INTERNLM = "internlm"
    
class QuantizationType(str, Enum):
    """量化类型"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP4 = "fp4"

class TrainingMode(str, Enum):
    """训练模式"""
    LORA = "lora"
    QLORA = "qlora"
    FULL_FINETUNING = "full"
    PREFIX_TUNING = "prefix"
    P_TUNING = "p_tuning"

@dataclass
class LoRAConfig:
    """LoRA配置参数"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    
@dataclass
class QuantizationConfig:
    """量化配置参数"""
    quantization_type: QuantizationType
    bits: int = 4
    use_double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_nested_quant: bool = False

@dataclass
class TrainingConfig:
    """训练配置参数"""
    model_name: str
    model_architecture: ModelArchitecture
    training_mode: TrainingMode
    dataset_path: str
    output_dir: str
    
    # 训练超参数
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # LoRA配置
    lora_config: Optional[LoRAConfig] = None
    
    # 量化配置  
    quantization_config: Optional[QuantizationConfig] = None
    
    # 分布式训练
    use_distributed: bool = False
    world_size: int = 1
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # 其他配置
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

class TrainingJob(TypedDict):
    """训练任务"""
    job_id: str
    job_name: str
    config: TrainingConfig
    status: str  # pending, running, completed, failed, cancelled
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float  # 0.0 - 1.0
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    best_loss: Optional[float]
    gpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    logs: List[str]
    error_message: Optional[str]

@dataclass
class ModelCheckpoint:
    """模型检查点"""
    checkpoint_id: str
    job_id: str
    epoch: int
    step: int
    loss: float
    eval_loss: Optional[float]
    model_path: str
    metrics: Dict[str, float]
    created_at: datetime
    size_bytes: int

class FineTuningResult(TypedDict):
    """微调结果"""
    job_id: str
    final_model_path: str
    best_checkpoint: ModelCheckpoint
    training_metrics: Dict[str, List[float]]
    evaluation_results: Dict[str, float]
    training_time: float
    total_steps: int
    final_loss: float
    convergence_achieved: bool
