"""
分布式训练器实现
支持PyTorch DDP、FSDP、DeepSpeed ZeRO等分布式训练策略
"""
import os
import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from .lora_trainer import LoRATrainer
from .models import TrainingConfig


class DistributedTrainer(LoRATrainer):
    """分布式训练器"""
    
    def __init__(self, config: TrainingConfig, monitor: Optional[object] = None):
        """
        初始化分布式训练器
        
        Args:
            config: 训练配置
            monitor: 训练监控器
        """
        # 启用分布式训练
        config.use_distributed = True
        
        super().__init__(config, monitor)
        
        # 初始化Accelerator
        self.accelerator = self._setup_accelerator()
        
        # 分布式相关属性
        self.world_size = self.accelerator.num_processes
        self.local_rank = self.accelerator.local_process_index
        self.is_main_process = self.accelerator.is_main_process
        
        self.logger.info(f"分布式训练初始化完成 - Rank: {self.local_rank}/{self.world_size}")
        
        if self.is_main_process:
            self.monitor.log_event("distributed_training_initialized", {
                "world_size": self.world_size,
                "local_rank": self.local_rank,
                "distributed_type": str(self.accelerator.distributed_type)
            })
    
    def _setup_accelerator(self) -> Accelerator:
        """设置Accelerator"""
        # DeepSpeed配置
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            deepspeed_plugin = DeepSpeedPlugin(
                config_file=self.config.deepspeed_config,
                zero_stage=self._get_deepspeed_zero_stage(),
                offload_optimizer_device="cpu" if self._should_offload_optimizer() else "none",
                offload_param_device="cpu" if self._should_offload_params() else "none"
            )
        
        # 创建Accelerator
        accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision="bf16" if self.config.bf16 else "fp16" if self.config.fp16 else "no",
            log_with="wandb" if self._wandb_available() else "tensorboard",
            project_dir=self.config.output_dir
        )
        
        # 设置随机种子
        set_seed(42)
        
        return accelerator
    
    def _get_deepspeed_zero_stage(self) -> int:
        """获取DeepSpeed ZeRO阶段"""
        if self.config.deepspeed_config:
            try:
                with open(self.config.deepspeed_config, 'r') as f:
                    config = json.load(f)
                return config.get("zero_optimization", {}).get("stage", 2)
            except Exception as e:
                self.logger.warning(f"无法解析DeepSpeed配置: {e}")
        return 2  # 默认Stage 2
    
    def _should_offload_optimizer(self) -> bool:
        """是否应该offload优化器"""
        # 基于GPU内存和模型大小的启发式判断
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return memory_gb < 24  # 24GB以下显存启用offload
        return True
    
    def _should_offload_params(self) -> bool:
        """是否应该offload参数"""
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return memory_gb < 16  # 16GB以下显存启用参数offload
        return True
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器（分布式版本）"""
        if self.is_main_process:
            self.logger.info(f"主进程加载模型: {self.config.model_name}")
        
        # 调用父类方法加载模型
        super().load_model_and_tokenizer()
        
        # 等待所有进程完成模型加载
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("所有进程模型加载完成")
    
    def setup_peft_model(self):
        """设置PEFT模型（分布式版本）"""
        super().setup_peft_model()
        
        # 使用Accelerator准备模型
        self.model = self.accelerator.prepare_model(self.model)
        
        if self.is_main_process:
            self.logger.info("PEFT模型分布式准备完成")
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        """创建分布式训练器"""
        from transformers import TrainingArguments
        from .distributed_training_args import DistributedTrainingArguments
        
        if self.is_main_process:
            self.logger.info("创建分布式训练器")
        
        # 分布式训练参数
        training_args = DistributedTrainingArguments(
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
            run_name=f"distributed-lora-{self.config.model_name.split('/')[-1]}",
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            
            # 分布式相关参数
            local_rank=self.local_rank,
            ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False,
            dataloader_num_workers=4,
            
            # DeepSpeed相关
            deepspeed=self.config.deepspeed_config if self.config.use_deepspeed else None,
            
            # 保存和日志控制
            save_on_each_node=False,
            logging_first_step=True,
            
            # 混合精度
            fp16_full_eval=self.config.fp16,
            bf16_full_eval=self.config.bf16,
            
            # 梯度裁剪
            max_grad_norm=1.0
        )
        
        # 数据收集器
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.config.fp16 or self.config.bf16 else None
        )
        
        # 创建分布式训练器
        trainer = DistributedCustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            monitor=self.monitor if self.is_main_process else None,
            accelerator=self.accelerator
        )
        
        return trainer
    
    def train(self) -> Dict[str, Any]:
        """执行分布式训练"""
        try:
            if self.is_main_process:
                self.logger.info("开始分布式LoRA训练")
                self.monitor.log_event("distributed_training_start", {
                    "world_size": self.world_size,
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
            if self.is_main_process:
                self.logger.info("开始分布式模型训练...")
            
            train_result = self.trainer.train()
            
            # 等待所有进程完成训练
            self.accelerator.wait_for_everyone()
            
            # 6. 保存模型（仅主进程）
            if self.is_main_process:
                self.logger.info("保存分布式训练模型...")
                self.trainer.save_model()
                self.trainer.save_state()
                
                # 保存训练配置
                config_path = os.path.join(self.config.output_dir, "distributed_training_config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config_to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # 7. 收集训练结果
            result = {
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_steps_per_second": train_result.metrics["train_steps_per_second"],
                "train_loss": train_result.metrics["train_loss"],
                "final_model_path": self.config.output_dir,
                "total_epochs": self.config.num_train_epochs,
                "total_steps": train_result.global_step,
                "world_size": self.world_size,
                "distributed_type": str(self.accelerator.distributed_type)
            }
            
            if self.is_main_process:
                self.logger.info("分布式训练完成!")
                self.monitor.log_event("distributed_training_complete", result)
            
            return result
            
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"分布式训练失败: {str(e)}")
                self.monitor.log_event("distributed_training_failed", {"error": str(e)})
            raise e
        finally:
            # 清理资源
            self.cleanup()
    
    def cleanup(self):
        """清理分布式训练资源"""
        try:
            if hasattr(self, 'accelerator'):
                self.accelerator.free_memory()
            
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
        except Exception as e:
            self.logger.warning(f"清理分布式资源时出错: {e}")
    
    def get_distributed_metrics(self) -> Dict[str, Any]:
        """获取分布式训练指标"""
        metrics = {}
        
        if torch.cuda.is_available():
            # GPU使用情况
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
            metrics["gpu_utilization"] = torch.cuda.utilization()
        
        # 分布式信息
        metrics.update({
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "distributed_type": str(self.accelerator.distributed_type),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps
        })
        
        return metrics


class DistributedCustomTrainer:
    """自定义分布式训练器"""
    
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, 
                 tokenizer, monitor, accelerator):
        from transformers import Trainer
        
        # 创建基础训练器
        self.base_trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        self.monitor = monitor
        self.accelerator = accelerator
        self.step_count = 0
    
    def train(self):
        """分布式训练"""
        # 准备训练组件
        self.base_trainer.model, self.base_trainer.optimizer, \
        self.base_trainer.train_dataloader, self.base_trainer.lr_scheduler = \
            self.accelerator.prepare(
                self.base_trainer.model,
                self.base_trainer.optimizer if hasattr(self.base_trainer, 'optimizer') else None,
                self.base_trainer.get_train_dataloader(),
                self.base_trainer.lr_scheduler if hasattr(self.base_trainer, 'lr_scheduler') else None
            )
        
        # 开始训练
        return self.base_trainer.train()
    
    def save_model(self):
        """保存模型"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # 获取原始模型（去除分布式包装）
            unwrapped_model = self.accelerator.unwrap_model(self.base_trainer.model)
            
            # 保存PEFT模型
            if hasattr(unwrapped_model, 'save_pretrained'):
                unwrapped_model.save_pretrained(self.base_trainer.args.output_dir)
            
            # 保存分词器
            if hasattr(self.base_trainer, 'tokenizer') and self.base_trainer.tokenizer:
                self.base_trainer.tokenizer.save_pretrained(self.base_trainer.args.output_dir)
    
    def save_state(self):
        """保存训练状态"""
        if self.accelerator.is_main_process:
            self.base_trainer.save_state()
    
    def log(self, logs: Dict[str, float]):
        """日志记录"""
        if self.monitor and self.accelerator.is_main_process:
            # 记录训练指标
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.monitor.log_metric(key, value)
            
            # 记录分布式指标
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.monitor.log_metric("gpu_memory_used_gb", memory_used)
                self.monitor.log_metric("gpu_memory_utilization", memory_used / memory_total * 100)
            
            self.step_count += 1


class DistributedTrainingArguments:
    """分布式训练参数"""
    
    def __init__(self, **kwargs):
        from transformers import TrainingArguments
        
        # 分布式特定的默认值
        distributed_defaults = {
            "ddp_find_unused_parameters": False,
            "ddp_broadcast_buffers": False,
            "dataloader_num_workers": 4,
            "save_on_each_node": False,
            "logging_first_step": True,
            "max_grad_norm": 1.0
        }
        
        # 合并默认值和用户参数
        all_kwargs = {**distributed_defaults, **kwargs}
        
        # 创建TrainingArguments实例
        self.training_args = TrainingArguments(**all_kwargs)
        
        # 代理所有属性
        for attr_name in dir(self.training_args):
            if not attr_name.startswith('_'):
                setattr(self, attr_name, getattr(self.training_args, attr_name))


def create_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    output_path: str = "deepspeed_config.json"
) -> str:
    """
    创建DeepSpeed配置文件
    
    Args:
        stage: ZeRO阶段 (1, 2, 3)
        offload_optimizer: 是否offload优化器到CPU
        offload_param: 是否offload参数到CPU
        output_path: 输出路径
        
    Returns:
        配置文件路径
    """
    config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    # CPU offload配置
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    if offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # 保存配置文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    return output_path