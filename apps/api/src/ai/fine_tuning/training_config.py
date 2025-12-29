"""
训练配置管理模块
负责配置文件解析、验证、预设模板管理等
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
from .models import TrainingConfig, LoRAConfig, QuantizationConfig, ModelArchitecture, TrainingMode, QuantizationType

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config_templates = self._load_config_templates()
        
    def _load_config_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载配置模板"""
        templates = {
            "lora_small": {
                "training_mode": TrainingMode.LORA,
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "lora_config": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj"]
                }
            },
            "lora_medium": {
                "training_mode": TrainingMode.LORA,
                "learning_rate": 1e-4,
                "num_train_epochs": 5,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lora_config": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                }
            },
            "lora_large": {
                "training_mode": TrainingMode.LORA,
                "learning_rate": 5e-5,
                "num_train_epochs": 8,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "lora_config": {
                    "rank": 32,
                    "alpha": 64,
                    "dropout": 0.05,
                    "target_modules": ["all-linear"]
                }
            },
            "qlora_4bit": {
                "training_mode": TrainingMode.QLORA,
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "lora_config": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["all-linear"]
                },
                "quantization_config": {
                    "quantization_type": QuantizationType.NF4,
                    "bits": 4,
                    "use_double_quant": True,
                    "compute_dtype": "bfloat16"
                }
            },
            "qlora_8bit": {
                "training_mode": TrainingMode.QLORA,
                "learning_rate": 1e-4,
                "num_train_epochs": 5,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "lora_config": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["all-linear"]
                },
                "quantization_config": {
                    "quantization_type": QuantizationType.INT8,
                    "bits": 8,
                    "compute_dtype": "bfloat16"
                }
            }
        }
        return templates
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """获取配置模板"""
        return self.config_templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """列出所有模板名称"""
        return list(self.config_templates.keys())
    
    def load_config_from_file(self, config_path: str) -> TrainingConfig:
        """从文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("不支持的配置文件格式，仅支持JSON和YAML")
        
        return self.dict_to_config(config_dict)
    
    def save_config_to_file(self, config: TrainingConfig, config_path: str):
        """保存配置到文件"""
        config_dict = self.config_to_dict(config)
        config_path = Path(config_path)
        
        os.makedirs(config_path.parent, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError("不支持的配置文件格式，仅支持JSON和YAML")
    
    def dict_to_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """将字典转换为TrainingConfig对象"""
        # 提取LoRA配置
        lora_config = None
        if 'lora_config' in config_dict:
            lora_dict = config_dict.pop('lora_config')
            lora_config = LoRAConfig(**lora_dict)
        
        # 提取量化配置
        quantization_config = None
        if 'quantization_config' in config_dict:
            quant_dict = config_dict.pop('quantization_config')
            quantization_config = QuantizationConfig(**quant_dict)
        
        # 转换枚举类型
        if 'model_architecture' in config_dict:
            config_dict['model_architecture'] = ModelArchitecture(config_dict['model_architecture'])
        if 'training_mode' in config_dict:
            config_dict['training_mode'] = TrainingMode(config_dict['training_mode'])
        
        # 创建TrainingConfig对象
        config = TrainingConfig(**config_dict)
        config.lora_config = lora_config
        config.quantization_config = quantization_config
        
        return config
    
    def config_to_dict(self, config: TrainingConfig) -> Dict[str, Any]:
        """将TrainingConfig对象转换为字典"""
        config_dict = {}
        
        # 基础配置
        for field, value in config.__dict__.items():
            if field in ['lora_config', 'quantization_config']:
                continue
            if hasattr(value, 'value'):  # 枚举类型
                config_dict[field] = value.value
            else:
                config_dict[field] = value
        
        # LoRA配置
        if config.lora_config:
            config_dict['lora_config'] = config.lora_config.__dict__
        
        # 量化配置
        if config.quantization_config:
            quant_dict = config.quantization_config.__dict__.copy()
            if 'quantization_type' in quant_dict:
                quant_dict['quantization_type'] = quant_dict['quantization_type'].value
            config_dict['quantization_config'] = quant_dict
        
        return config_dict
    
    def create_config_from_template(self, template_name: str, **overrides) -> TrainingConfig:
        """从模板创建配置"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"模板不存在: {template_name}")
        
        # 合并模板和覆盖参数
        config_dict = {**template, **overrides}
        
        return self.dict_to_config(config_dict)
    
    def validate_config(self, config: TrainingConfig) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 检查必需参数
        if not config.model_name:
            errors.append("model_name 不能为空")
        if not config.dataset_path:
            errors.append("dataset_path 不能为空")
        if not config.output_dir:
            errors.append("output_dir 不能为空")
        
        # 检查超参数范围
        if config.learning_rate <= 0 or config.learning_rate > 1:
            errors.append("learning_rate 应该在 (0, 1] 范围内")
        if config.num_train_epochs <= 0:
            errors.append("num_train_epochs 应该大于 0")
        if config.per_device_train_batch_size <= 0:
            errors.append("per_device_train_batch_size 应该大于 0")
        if config.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps 应该大于 0")
        
        # 检查LoRA配置
        if config.lora_config:
            if config.lora_config.rank <= 0:
                errors.append("LoRA rank 应该大于 0")
            if config.lora_config.alpha <= 0:
                errors.append("LoRA alpha 应该大于 0")
            if not (0 <= config.lora_config.dropout < 1):
                errors.append("LoRA dropout 应该在 [0, 1) 范围内")
        
        # 检查量化配置
        if config.quantization_config:
            if config.quantization_config.bits not in [4, 8]:
                errors.append("量化位数应该为 4 或 8")
        
        # 检查文件路径
        if config.dataset_path and not config.dataset_path.startswith('http'):
            if not os.path.exists(config.dataset_path):
                errors.append(f"数据集路径不存在: {config.dataset_path}")
        
        return errors
    
    def detect_hardware_config(self) -> Dict[str, Any]:
        """检测硬件配置并推荐参数"""
        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'total_memory': [],
            'device_names': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                hardware_info['device_names'].append(props.name)
                hardware_info['total_memory'].append(props.total_memory // (1024**3))  # GB
        
        # 基于硬件推荐配置
        recommendations = {
            'use_bf16': torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'use_fp16': torch.cuda.is_available(),
            'use_gradient_checkpointing': True,
            'use_flash_attention': torch.cuda.is_available()
        }
        
        # 根据显存推荐批次大小
        if hardware_info['total_memory']:
            max_memory = max(hardware_info['total_memory'])
            if max_memory >= 40:
                recommendations['per_device_train_batch_size'] = 8
            elif max_memory >= 24:
                recommendations['per_device_train_batch_size'] = 4
            elif max_memory >= 16:
                recommendations['per_device_train_batch_size'] = 2
            else:
                recommendations['per_device_train_batch_size'] = 1
                recommendations['gradient_accumulation_steps'] = 16
        
        return {
            'hardware_info': hardware_info,
            'recommendations': recommendations
        }
    
    def get_model_specific_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型特定的配置建议"""
        model_configs = {
            'llama': {
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'max_seq_length': 4096
            },
            'mistral': {
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'max_seq_length': 8192
            },
            'qwen': {
                'target_modules': ["c_attn", "c_proj", "w1", "w2"],
                'max_seq_length': 8192
            },
            'chatglm': {
                'target_modules': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
                'max_seq_length': 8192
            }
        }
        
        # 根据模型名称猜测架构
        model_name_lower = model_name.lower()
        for arch, config in model_configs.items():
            if arch in model_name_lower:
                return config
        
        # 默认配置
        return {
            'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
            'max_seq_length': 2048
        }
