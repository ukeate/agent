"""
LoRA训练器实现
基于Hugging Face PEFT库实现高效的LoRA微调
"""
import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from .models import TrainingConfig, ModelArchitecture, TrainingMode, QuantizationType
from .training_monitor import TrainingMonitor


class LoRATrainer:
    """LoRA微调训练器"""
    
    def __init__(self, config: TrainingConfig, monitor: Optional[TrainingMonitor] = None):
        """
        初始化LoRA训练器
        
        Args:
            config: 训练配置
            monitor: 训练监控器
        """
        self.config = config
        self.monitor = monitor or TrainingMonitor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_model = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 初始化训练环境
        self._setup_environment()
        
        # 记录开始训练
        self.monitor.log_event("trainer_initialized", {
            "model_name": self.config.model_name,
            "training_mode": self.config.training_mode.value,
            "output_dir": self.config.output_dir
        })
        
    def _setup_environment(self):
        """设置训练环境"""
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            self.monitor.log_metric("gpu_count", torch.cuda.device_count())
            self.monitor.log_metric("gpu_memory_total", torch.cuda.get_device_properties(0).total_memory / 1024**3)
        else:
            self.device = torch.device("cpu")
            self.logger.info("使用CPU")
        
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.logger.info(f"加载模型: {self.config.model_name}")
        self.monitor.log_event("model_loading_start", {"model_name": self.config.model_name})
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 量化配置
            quantization_config = None
            if self.config.quantization_config:
                quantization_config = self._create_quantization_config()
                self.logger.info(f"启用量化: {self.config.quantization_config.quantization_type.value}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if quantization_config else None,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16 if self.config.fp16 else torch.float32,
                trust_remote_code=True,
                use_flash_attention_2=self.config.use_flash_attention
            )
            
            # 设置梯度检查点
            if self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                self.logger.info("启用梯度检查点")
                
            # 如果使用量化，准备模型
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
                
            self.logger.info("模型和分词器加载成功")
            self.monitor.log_event("model_loading_complete", {
                "model_parameters": self.model.num_parameters(),
                "model_dtype": str(self.model.dtype)
            })
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            self.monitor.log_event("model_loading_failed", {"error": str(e)})
            raise e
    
    def _create_quantization_config(self):
        """创建量化配置"""
        from transformers import BitsAndBytesConfig
        
        qconfig = self.config.quantization_config
        
        if qconfig.quantization_type == QuantizationType.INT4 or qconfig.quantization_type == QuantizationType.NF4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, qconfig.compute_dtype),
                bnb_4bit_use_double_quant=qconfig.use_double_quant,
                bnb_4bit_quant_type=qconfig.quant_type if qconfig.quantization_type == QuantizationType.NF4 else "fp4"
            )
        elif qconfig.quantization_type == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=getattr(torch, qconfig.compute_dtype)
            )
        
        return None
    
    def setup_peft_model(self):
        """设置PEFT模型"""
        if self.config.training_mode in [TrainingMode.LORA, TrainingMode.QLORA]:
            self.logger.info("设置LoRA配置")
            
            # 获取目标模块
            target_modules = self._get_target_modules()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_config.rank,
                lora_alpha=self.config.lora_config.alpha,
                lora_dropout=self.config.lora_config.dropout,
                target_modules=target_modules,
                bias=self.config.lora_config.bias
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # 打印可训练参数
            trainable_params = self.peft_model.num_parameters(only_trainable=True)
            total_params = self.peft_model.num_parameters()
            
            self.logger.info(
                f"可训练参数: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
            
            self.monitor.log_metric("trainable_parameters", trainable_params)
            self.monitor.log_metric("total_parameters", total_params)
            self.monitor.log_metric("trainable_percentage", 100 * trainable_params / total_params)
            
            # 更新模型引用
            self.model = self.peft_model
    
    def _get_target_modules(self) -> Union[List[str], str]:
        """获取目标模块"""
        if self.config.lora_config.target_modules:
            return self.config.lora_config.target_modules
        
        # 根据模型架构返回适当的目标模块
        arch_mapping = {
            ModelArchitecture.LLAMA: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelArchitecture.MISTRAL: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ModelArchitecture.QWEN: ["c_attn", "c_proj", "w1", "w2"],
            ModelArchitecture.CHATGLM: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }
        
        return arch_mapping.get(
            self.config.model_architecture, 
            ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
    
    def prepare_dataset(self, dataset_path: str) -> Dataset:
        """准备训练数据集"""
        self.logger.info(f"加载数据集: {dataset_path}")
        self.monitor.log_event("dataset_loading_start", {"dataset_path": dataset_path})
        
        try:
            # 加载数据集
            if dataset_path.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            else:
                dataset = load_dataset(dataset_path, split='train')
            
            self.logger.info(f"数据集大小: {len(dataset)}")
            self.monitor.log_metric("dataset_size", len(dataset))
            
            # 数据预处理
            def tokenize_function(examples):
                # 构建输入文本
                texts = []
                for i in range(len(examples['instruction'])):
                    instruction = examples['instruction'][i]
                    output = examples['output'][i]
                    text = f"### Instruction:\n{instruction}\n### Response:\n{output}{self.tokenizer.eos_token}"
                    texts.append(text)
                
                # 分词
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False
                )
                
                # 设置labels
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="分词处理中"
            )
            
            self.monitor.log_event("dataset_processing_complete", {
                "tokenized_size": len(tokenized_dataset),
                "max_seq_length": self.config.max_seq_length
            })
            
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"数据集处理失败: {str(e)}")
            self.monitor.log_event("dataset_processing_failed", {"error": str(e)})
            raise e
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Trainer:
        """创建训练器"""
        self.logger.info("创建训练器")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            report_to="wandb" if self._wandb_available() else "tensorboard",
            run_name=f"lora-{self.config.model_name.split('/')[-1]}",
            logging_dir=os.path.join(self.config.output_dir, "logs")
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.config.fp16 or self.config.bf16 else None
        )
        
        # 创建自定义训练器
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            monitor=self.monitor
        )
        
        return trainer
    
    def train(self) -> Dict[str, Any]:
        """执行训练"""
        try:
            self.logger.info("开始LoRA训练")
            self.monitor.log_event("training_start", {
                "epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size,
                "learning_rate": self.config.learning_rate
            })
            
            # 1. 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 2. 设置PEFT模型
            self.setup_peft_model()
            
            # 3. 准备数据集
            train_dataset = self.prepare_dataset(self.config.dataset_path)
            
            # 4. 创建训练器
            self.trainer = self.create_trainer(train_dataset)
            
            # 5. 开始训练
            self.logger.info("开始模型训练...")
            train_result = self.trainer.train()
            
            # 6. 保存模型
            self.logger.info("保存模型...")
            self.trainer.save_model()
            self.trainer.save_state()
            
            # 7. 保存训练配置
            config_path = os.path.join(self.config.output_dir, "training_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config_to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # 8. 保存训练结果
            result = {
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_steps_per_second": train_result.metrics["train_steps_per_second"],
                "train_loss": train_result.metrics["train_loss"],
                "final_model_path": self.config.output_dir,
                "total_epochs": self.config.num_train_epochs,
                "total_steps": train_result.global_step,
                "config_path": config_path
            }
            
            self.logger.info("训练完成!")
            self.monitor.log_event("training_complete", result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"训练失败: {str(e)}")
            self.monitor.log_event("training_failed", {"error": str(e)})
            raise e
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            elif hasattr(value, 'value'):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    def _wandb_available(self) -> bool:
        """检查wandb是否可用"""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """评估模型"""
        if not self.trainer:
            raise ValueError("训练器尚未初始化，请先调用train()方法")
        
        if eval_dataset is None and not hasattr(self.trainer, 'eval_dataset'):
            raise ValueError("未提供评估数据集")
        
        self.logger.info("开始模型评估")
        
        if eval_dataset:
            # 临时设置评估数据集
            original_eval_dataset = self.trainer.eval_dataset
            self.trainer.eval_dataset = eval_dataset
            eval_result = self.trainer.evaluate()
            self.trainer.eval_dataset = original_eval_dataset
        else:
            eval_result = self.trainer.evaluate()
        
        self.logger.info(f"评估完成: {eval_result}")
        return eval_result
    
    def save_model(self, save_directory: str):
        """保存模型到指定目录"""
        if not self.peft_model:
            raise ValueError("PEFT模型尚未初始化")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存PEFT模型
        self.peft_model.save_pretrained(save_directory)
        
        # 保存分词器
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)
        
        self.logger.info(f"模型已保存到: {save_directory}")
    
    def load_model(self, model_directory: str):
        """从目录加载模型"""
        from peft import PeftModel
        
        self.logger.info(f"从目录加载模型: {model_directory}")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto"
        )
        
        # 加载PEFT模型
        self.peft_model = PeftModel.from_pretrained(base_model, model_directory)
        self.model = self.peft_model
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        self.logger.info("模型加载完成")


class CustomTrainer(Trainer):
    """自定义训练器，增加监控功能"""
    
    def __init__(self, *args, monitor: TrainingMonitor, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.step_count = 0
        
    def log(self, logs: Dict[str, float]) -> None:
        """重写日志记录方法"""
        super().log(logs)
        
        # 记录训练指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.monitor.log_metric(key, value)
        
        # 记录GPU使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.monitor.log_metric("gpu_memory_used_gb", memory_used)
            self.monitor.log_metric("gpu_memory_utilization", memory_used / memory_total * 100)
        
        self.step_count += 1
    
    def on_epoch_begin(self):
        """epoch开始时的回调"""
        super().on_epoch_begin() if hasattr(super(), 'on_epoch_begin') else None
        self.monitor.log_event("epoch_start", {"epoch": self.state.epoch})