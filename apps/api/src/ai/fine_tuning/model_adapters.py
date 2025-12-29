"""
模型适配器模块
支持不同模型架构的自动适配和优化
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from .models import ModelArchitecture

from src.core.logging import get_logger
logger = get_logger(__name__)

class BaseModelAdapter(ABC):
    """模型适配器基类"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def get_target_modules(self) -> List[str]:
        """获取目标模块列表"""
        raise NotImplementedError
    
    @abstractmethod
    def get_architecture(self) -> ModelArchitecture:
        """获取模型架构"""
        raise NotImplementedError
    
    @abstractmethod
    def get_max_sequence_length(self) -> int:
        """获取最大序列长度"""
        raise NotImplementedError
    
    @abstractmethod
    def get_optimization_config(self) -> Dict[str, Any]:
        """获取优化配置"""
        raise NotImplementedError
    
    def get_attention_implementation(self) -> str:
        """获取注意力实现方式"""
        return "sdpa"  # 默认使用scaled_dot_product_attention
    
    def get_rope_scaling(self) -> Optional[Dict[str, Any]]:
        """获取RoPE缩放配置"""
        return None
    
    def requires_special_tokenizer_config(self) -> Dict[str, Any]:
        """是否需要特殊的分词器配置"""
        return {}

class LlamaAdapter(BaseModelAdapter):
    """LLaMA模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.LLAMA
    
    def get_max_sequence_length(self) -> int:
        # 根据模型版本确定
        if "llama-2" in self.model_name.lower():
            return 4096
        elif "llama-3" in self.model_name.lower():
            return 8192
        else:
            return 4096  # 默认值
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "rope_theta": 10000.0,
            "attention_bias": False
        }
    
    def get_attention_implementation(self) -> str:
        return "flash_attention_2"
    
    def get_rope_scaling(self) -> Optional[Dict[str, Any]]:
        # LLaMA-3.1支持更长的上下文
        if "llama-3.1" in self.model_name.lower():
            return {
                "type": "linear",
                "factor": 8.0
            }
        return None

class MistralAdapter(BaseModelAdapter):
    """Mistral模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.MISTRAL
    
    def get_max_sequence_length(self) -> int:
        if "mistral-7b" in self.model_name.lower():
            return 8192
        elif "mistral-8x7b" in self.model_name.lower():  # Mixtral
            return 32768
        else:
            return 8192
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "sliding_window": 4096,  # Mistral特有的滑动窗口
            "attention_bias": False
        }
    
    def get_attention_implementation(self) -> str:
        return "flash_attention_2"

class QwenAdapter(BaseModelAdapter):
    """Qwen模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        # Qwen使用不同的模块命名
        if "qwen2" in self.model_name.lower():
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        else:
            return [
                "c_attn", "c_proj",
                "w1", "w2"
            ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.QWEN
    
    def get_max_sequence_length(self) -> int:
        if "qwen2" in self.model_name.lower():
            return 32768  # Qwen2支持更长上下文
        else:
            return 8192
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": True,
            "use_dynamic_ntk": True  # Qwen特有的动态NTK
        }
    
    def requires_special_tokenizer_config(self) -> Dict[str, Any]:
        return {
            "trust_remote_code": True,
            "padding_side": "left"  # Qwen推荐左填充
        }

class ChatGLMAdapter(BaseModelAdapter):
    """ChatGLM模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        if "chatglm3" in self.model_name.lower():
            return [
                "query_key_value", "dense",
                "dense_h_to_4h", "dense_4h_to_h"
            ]
        else:
            return [
                "query_key_value", "dense",
                "dense_h_to_4h", "dense_4h_to_h"
            ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.CHATGLM
    
    def get_max_sequence_length(self) -> int:
        if "chatglm3" in self.model_name.lower():
            return 8192
        else:
            return 8192
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": True,
            "pre_seq_len": None  # ChatGLM的prefix tuning
        }
    
    def requires_special_tokenizer_config(self) -> Dict[str, Any]:
        return {
            "trust_remote_code": True,
            "padding_side": "left"
        }

class BaichuanAdapter(BaseModelAdapter):
    """Baichuan模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "W_pack", "o_proj",  # Baichuan特有的打包权重
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.BAICHUAN
    
    def get_max_sequence_length(self) -> int:
        if "baichuan2" in self.model_name.lower():
            return 4096
        else:
            return 4096
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": False,
            "rope_theta": 10000.0
        }
    
    def requires_special_tokenizer_config(self) -> Dict[str, Any]:
        return {
            "trust_remote_code": True,
            "use_fast": False  # Baichuan可能需要使用慢速分词器
        }

class YiAdapter(BaseModelAdapter):
    """Yi模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.YI
    
    def get_max_sequence_length(self) -> int:
        if "yi-34b" in self.model_name.lower():
            return 4096
        elif "yi-6b" in self.model_name.lower():
            return 4096
        else:
            return 4096
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": False,
            "rope_theta": 5000000.0  # Yi使用不同的RoPE theta
        }

class DeepSeekAdapter(BaseModelAdapter):
    """DeepSeek模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.DEEPSEEK
    
    def get_max_sequence_length(self) -> int:
        return 4096
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": False
        }

class InternLMAdapter(BaseModelAdapter):
    """InternLM模型适配器"""
    
    def get_target_modules(self) -> List[str]:
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture.INTERNLM
    
    def get_max_sequence_length(self) -> int:
        if "internlm2" in self.model_name.lower():
            return 8192
        else:
            return 4096
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "use_flash_attention": True,
            "use_gradient_checkpointing": True,
            "attention_bias": False
        }

class ModelAdapterFactory:
    """模型适配器工厂类"""
    
    # 模型名称模式到适配器的映射
    ADAPTER_MAPPING = {
        r'llama': LlamaAdapter,
        r'mistral|mixtral': MistralAdapter,
        r'qwen': QwenAdapter,
        r'chatglm': ChatGLMAdapter,
        r'baichuan': BaichuanAdapter,
        r'yi-\d+b': YiAdapter,
        r'deepseek': DeepSeekAdapter,
        r'internlm': InternLMAdapter,
    }
    
    @classmethod
    def create_adapter(cls, model_name: str) -> BaseModelAdapter:
        """
        根据模型名称创建适配器
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型适配器实例
        """
        model_name_lower = model_name.lower()
        
        # 遍历映射，找到匹配的模式
        for pattern, adapter_class in cls.ADAPTER_MAPPING.items():
            if re.search(pattern, model_name_lower):
                return adapter_class(model_name)
        
        # 如果没有匹配的，使用默认的LLaMA适配器
        logger.warning(f"未找到模型 {model_name} 的专用适配器，使用默认LLaMA适配器")
        return LlamaAdapter(model_name)
    
    @classmethod
    def get_supported_architectures(cls) -> List[str]:
        """获取支持的模型架构列表"""
        return [arch.value for arch in ModelArchitecture]
    
    @classmethod
    def detect_model_architecture(cls, model_name: str) -> ModelArchitecture:
        """
        检测模型架构
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型架构枚举
        """
        adapter = cls.create_adapter(model_name)
        return adapter.get_architecture()

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, adapter: BaseModelAdapter):
        self.adapter = adapter
        self.logger = get_logger(self.__class__.__name__)
    
    def get_recommended_batch_size(self, gpu_memory_gb: float, sequence_length: int) -> Tuple[int, int]:
        """
        根据GPU内存和序列长度推荐批次大小
        
        Args:
            gpu_memory_gb: GPU内存大小(GB)
            sequence_length: 序列长度
            
        Returns:
            (per_device_batch_size, gradient_accumulation_steps)
        """
        # 简化的内存估算公式
        # 考虑模型大小、序列长度、激活值等
        
        # 基础内存需求（GB）
        base_memory = 2.0
        
        # 序列长度相关的内存需求
        seq_memory = sequence_length * 0.001  # 简化估算
        
        # 可用内存（预留20%给系统）
        available_memory = gpu_memory_gb * 0.8 - base_memory
        
        # 每个样本的内存需求
        per_sample_memory = seq_memory + 0.5  # 梯度、优化器状态等
        
        # 计算批次大小
        max_batch_size = max(1, int(available_memory / per_sample_memory))
        
        # 根据模型架构调整
        if self.adapter.get_architecture() == ModelArchitecture.MISTRAL:
            max_batch_size = max(1, max_batch_size // 2)  # Mistral需要更多内存
        
        # 确定最优的批次大小和梯度累积步数
        if max_batch_size >= 8:
            return 8, 1
        elif max_batch_size >= 4:
            return 4, 2
        elif max_batch_size >= 2:
            return 2, 4
        else:
            return 1, 8
    
    def get_learning_rate_schedule(self, total_steps: int) -> Dict[str, Any]:
        """
        获取学习率调度配置
        
        Args:
            total_steps: 总训练步数
            
        Returns:
            学习率调度配置
        """
        warmup_steps = min(total_steps // 10, 1000)  # 10%的步数或最多1000步
        
        schedule_config = {
            "scheduler_type": "cosine",
            "warmup_steps": warmup_steps,
            "warmup_ratio": warmup_steps / total_steps,
            "num_training_steps": total_steps
        }
        
        return schedule_config
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        config = {
            "optimizer_type": "adamw_torch",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8
        }
        
        # 根据模型架构调整
        arch = self.adapter.get_architecture()
        if arch == ModelArchitecture.QWEN:
            config["learning_rate"] = 1e-4  # Qwen通常使用较小的学习率
        elif arch == ModelArchitecture.CHATGLM:
            config["weight_decay"] = 0.1  # ChatGLM使用较大的权重衰减
        
        return config
    
    def get_quantization_recommendations(self, gpu_memory_gb: float) -> Dict[str, Any]:
        """
        获取量化推荐
        
        Args:
            gpu_memory_gb: GPU内存大小
            
        Returns:
            量化推荐配置
        """
        recommendations = {}
        
        if gpu_memory_gb < 12:
            # 低显存：推荐4-bit量化
            recommendations = {
                "use_quantization": True,
                "quantization_type": "nf4",
                "bits": 4,
                "use_double_quant": True,
                "compute_dtype": "bfloat16"
            }
        elif gpu_memory_gb < 24:
            # 中等显存：推荐8-bit量化
            recommendations = {
                "use_quantization": True,
                "quantization_type": "int8",
                "bits": 8,
                "compute_dtype": "bfloat16"
            }
        else:
            # 高显存：不使用量化
            recommendations = {
                "use_quantization": False,
                "use_bf16": True
            }
        
        return recommendations
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置的兼容性
        
        Args:
            config: 训练配置
            
        Returns:
            警告和错误列表
        """
        warnings = []
        
        arch = self.adapter.get_architecture()
        max_seq_len = self.adapter.get_max_sequence_length()
        
        # 检查序列长度
        if config.get("max_seq_length", 0) > max_seq_len:
            warnings.append(
                f"序列长度 {config['max_seq_length']} 超过模型最大支持长度 {max_seq_len}"
            )
        
        # 检查注意力实现
        flash_attention = config.get("use_flash_attention", False)
        if flash_attention and arch in [ModelArchitecture.CHATGLM]:
            warnings.append(
                f"{arch.value} 模型可能不完全支持Flash Attention"
            )
        
        # 检查量化配置
        quantization = config.get("quantization_config")
        if quantization and quantization.get("bits") == 4:
            if arch in [ModelArchitecture.CHATGLM]:
                warnings.append(
                    f"{arch.value} 模型的4-bit量化支持可能不稳定"
                )
        
        return warnings
