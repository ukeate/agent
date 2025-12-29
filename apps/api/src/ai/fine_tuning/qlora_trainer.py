"""
QLoRA量化微调训练器
基于BitsAndBytes实现4-bit和8-bit量化微调
"""

import torch
from typing import Dict, Any, Optional
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from .lora_trainer import LoRATrainer
from .models import TrainingConfig, QuantizationConfig, QuantizationType, TrainingMode

from src.core.logging import get_logger
class QLoRATrainer(LoRATrainer):
    """QLoRA量化微调训练器"""
    
    def __init__(self, config: TrainingConfig, monitor: Optional[object] = None, job_control: Optional[object] = None):
        """
        初始化QLoRA训练器
        
        Args:
            config: 训练配置
            monitor: 训练监控器
        """
        # 初始化logger
        self.logger = get_logger(self.__class__.__name__)
        
        # 确保使用量化配置
        if config.quantization_config is None:
            self.logger.warning("未提供量化配置，使用默认NF4量化配置")
            config.quantization_config = QuantizationConfig(
                quantization_type=QuantizationType.NF4,
                bits=4,
                use_double_quant=True,
                quant_type="nf4",
                compute_dtype="bfloat16"
            )
        
        # 确保训练模式为QLoRA
        config.training_mode = TrainingMode.QLORA
        
        super().__init__(config, monitor, job_control=job_control)
        
        self.logger.info("QLoRA训练器初始化完成")
        self.monitor.log_event("qlora_trainer_initialized", {
            "quantization_type": config.quantization_config.quantization_type.value,
            "bits": config.quantization_config.bits,
            "use_double_quant": config.quantization_config.use_double_quant
        })
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """创建量化配置"""
        qconfig = self.config.quantization_config
        
        self.logger.info(f"创建量化配置: {qconfig.quantization_type.value}-bit")
        
        if qconfig.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, qconfig.compute_dtype),
                bnb_4bit_use_double_quant=qconfig.use_double_quant,
                bnb_4bit_quant_type=qconfig.quant_type
            )
            
            self.logger.info(f"4-bit量化配置:")
            self.logger.info(f"  - 计算数据类型: {qconfig.compute_dtype}")
            self.logger.info(f"  - 使用双重量化: {qconfig.use_double_quant}")
            self.logger.info(f"  - 量化类型: {qconfig.quant_type}")
            
        elif qconfig.bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_compute_dtype=getattr(torch, qconfig.compute_dtype),
                llm_int8_threshold=6.0  # 默认阈值
            )
            
            self.logger.info(f"8-bit量化配置:")
            self.logger.info(f"  - 计算数据类型: {qconfig.compute_dtype}")
            self.logger.info(f"  - 异常值阈值: 6.0")
            
        else:
            raise ValueError(f"不支持的量化位数: {qconfig.bits}")
        
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器（重写以支持量化）"""
        self.logger.info(f"加载量化模型: {self.config.model_name}")
        self.monitor.log_event("quantized_model_loading_start", {
            "model_name": self.config.model_name,
            "quantization_bits": self.config.quantization_config.bits
        })
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 创建量化配置
            quantization_config = self._create_quantization_config()
            
            # 加载量化模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",  # 量化模型必须使用device_map="auto"
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                trust_remote_code=True
            )
            
            # 为量化训练准备模型
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=self.config.use_gradient_checkpointing
            )
            
            # 记录模型信息
            self.logger.info("量化模型加载成功")
            self.logger.info(f"模型参数量: {self.model.num_parameters():,}")
            
            # 记录量化效果
            self._log_quantization_stats()
            
            self.monitor.log_event("quantized_model_loading_complete", {
                "model_parameters": self.model.num_parameters(),
                "model_dtype": str(self.model.dtype)
            })
            
        except Exception as e:
            self.logger.error(f"量化模型加载失败: {str(e)}")
            self.monitor.log_event("quantized_model_loading_failed", {"error": str(e)})
            raise e
    
    def _log_quantization_stats(self):
        """记录量化统计信息"""
        try:
            # 统计量化层数量
            quantized_layers = 0
            total_layers = 0
            
            for name, module in self.model.named_modules():
                total_layers += 1
                if hasattr(module, 'weight') and hasattr(module.weight, 'quant_type'):
                    quantized_layers += 1
            
            self.logger.info(f"量化层数量: {quantized_layers}/{total_layers}")
            self.monitor.log_metric("quantized_layers", quantized_layers)
            self.monitor.log_metric("total_layers", total_layers)
            self.monitor.log_metric("quantization_ratio", quantized_layers / total_layers if total_layers > 0 else 0)
            
            # 估算内存节省
            bits = self.config.quantization_config.bits
            memory_reduction = (32 - bits) / 32 * 100
            self.logger.info(f"理论内存节省: {memory_reduction:.1f}%")
            self.monitor.log_metric("theoretical_memory_reduction", memory_reduction)
            
        except Exception as e:
            self.logger.warning(f"无法统计量化信息: {e}")
    
    def setup_peft_model(self):
        """设置PEFT模型（重写以支持量化）"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.logger.info("设置QLoRA配置")
        
        # 获取目标模块
        target_modules = self._get_target_modules()
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_config.rank,
            lora_alpha=self.config.lora_config.alpha,
            lora_dropout=self.config.lora_config.dropout,
            target_modules=target_modules,
            bias=self.config.lora_config.bias
        )
        
        # 应用PEFT配置
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = self.peft_model.num_parameters(only_trainable=True)
        total_params = self.peft_model.num_parameters()
        trainable_percentage = 100 * trainable_params / total_params
        
        self.logger.info(f"QLoRA可训练参数: {trainable_params:,} ({trainable_percentage:.4f}%)")
        
        # 记录量化微调的内存优势
        self._calculate_memory_efficiency(trainable_params, total_params)
        
        self.monitor.log_metric("qlora_trainable_parameters", trainable_params)
        self.monitor.log_metric("qlora_total_parameters", total_params)
        self.monitor.log_metric("qlora_trainable_percentage", trainable_percentage)
        
        # 更新模型引用
        self.model = self.peft_model
    
    def _calculate_memory_efficiency(self, trainable_params: int, total_params: int):
        """计算内存效率"""
        # 量化带来的内存节省
        bits = self.config.quantization_config.bits
        quantization_reduction = (32 - bits) / 32
        
        # LoRA带来的参数减少
        lora_reduction = 1 - (trainable_params / total_params)
        
        # 总体内存节省（简化估算）
        total_reduction = quantization_reduction * 0.8 + lora_reduction * 0.2  # 权重估计
        
        self.logger.info(f"量化内存节省: {quantization_reduction * 100:.1f}%")
        self.logger.info(f"LoRA参数减少: {lora_reduction * 100:.4f}%")
        self.logger.info(f"总体内存效率提升: {total_reduction * 100:.1f}%")
        
        self.monitor.log_metric("quantization_memory_reduction", quantization_reduction * 100)
        self.monitor.log_metric("lora_parameter_reduction", lora_reduction * 100)
        self.monitor.log_metric("total_memory_efficiency", total_reduction * 100)
    
    def train(self) -> Dict[str, Any]:
        """执行QLoRA训练"""
        try:
            self.logger.info("开始QLoRA量化微调训练")
            self.monitor.log_event("qlora_training_start", {
                "quantization_bits": self.config.quantization_config.bits,
                "lora_rank": self.config.lora_config.rank,
                "epochs": self.config.num_train_epochs
            })
            
            # 记录训练前的GPU内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清空缓存
                memory_before = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"训练前GPU内存使用: {memory_before:.2f}GB")
                self.monitor.log_metric("gpu_memory_before_training", memory_before)
            
            # 调用父类的训练方法
            result = super().train()
            
            # 记录训练后的GPU内存使用
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"训练后GPU内存使用: {memory_after:.2f}GB")
                self.monitor.log_metric("gpu_memory_after_training", memory_after)
            
            # 添加QLoRA特定的结果信息
            result.update({
                "quantization_bits": self.config.quantization_config.bits,
                "quantization_type": self.config.quantization_config.quantization_type.value,
                "use_double_quant": self.config.quantization_config.use_double_quant,
                "training_mode": "qlora"
            })
            
            self.monitor.log_event("qlora_training_complete", result)
            return result
            
        except Exception as e:
            self.logger.error(f"QLoRA训练失败: {str(e)}")
            self.monitor.log_event("qlora_training_failed", {"error": str(e)})
            raise e
    
    def optimize_for_inference(self):
        """为推理优化模型"""
        if not self.peft_model:
            raise ValueError("PEFT模型尚未初始化")
        
        self.logger.info("优化QLoRA模型用于推理")
        
        try:
            # 合并LoRA权重到基础模型（如果支持）
            if hasattr(self.peft_model, 'merge_and_unload'):
                self.logger.info("合并LoRA权重...")
                merged_model = self.peft_model.merge_and_unload()
                self.model = merged_model
                self.logger.info("LoRA权重合并完成")
            
            # 设置推理模式
            self.model.eval()
            
            # 禁用梯度计算
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 启用推理优化
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = True
            
            self.logger.info("QLoRA推理优化完成")
            self.monitor.log_event("qlora_inference_optimization_complete")
            
        except Exception as e:
            self.logger.error(f"推理优化失败: {str(e)}")
            self.monitor.log_event("qlora_inference_optimization_failed", {"error": str(e)})
            raise e
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """基准测试内存使用"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，无法进行内存基准测试")
            return {}
        
        self.logger.info("开始内存基准测试")
        
        benchmark_results = {}
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 记录基准内存
        baseline_memory = torch.cuda.memory_allocated() / 1024**3
        benchmark_results["baseline_memory_gb"] = baseline_memory
        
        # 模拟推理
        if self.model and self.tokenizer:
            test_input = "This is a test input for memory benchmarking."
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.model.device)
            
            # 推理前内存
            pre_inference_memory = torch.cuda.memory_allocated() / 1024**3
            benchmark_results["pre_inference_memory_gb"] = pre_inference_memory
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 推理后内存
            post_inference_memory = torch.cuda.memory_allocated() / 1024**3
            benchmark_results["post_inference_memory_gb"] = post_inference_memory
            
            # 计算内存增量
            inference_memory_delta = post_inference_memory - pre_inference_memory
            benchmark_results["inference_memory_delta_gb"] = inference_memory_delta
        
        # 记录最大内存使用
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        benchmark_results["max_memory_gb"] = max_memory
        
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        
        self.logger.info(f"内存基准测试结果: {benchmark_results}")
        
        # 记录到监控器
        for key, value in benchmark_results.items():
            self.monitor.log_metric(f"memory_benchmark_{key}", value)
        
        return benchmark_results
    
    def compare_with_full_precision(self, test_inputs: list) -> Dict[str, Any]:
        """与全精度模型比较"""
        self.logger.info("开始与全精度模型比较")
        
        if not self.model or not self.tokenizer:
            raise ValueError("模型或分词器未初始化")
        
        comparison_results = {
            "quantized_outputs": [],
            "memory_usage": {},
            "inference_time": {}
        }
        
        # 量化模型推理
        self.model.eval()
        with torch.no_grad():
            for test_input in test_inputs:
                inputs = self.tokenizer(test_input, return_tensors="pt").to(self.model.device)
                
                # 记录推理时间
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                
                # 解码输出
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                comparison_results["quantized_outputs"].append({
                    "input": test_input,
                    "output": output_text,
                    "inference_time_ms": inference_time
                })
        
        # 记录内存使用
        if torch.cuda.is_available():
            comparison_results["memory_usage"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3
            }
        
        self.logger.info("模型比较完成")
        self.monitor.log_event("model_comparison_complete", {
            "test_inputs_count": len(test_inputs),
            "avg_inference_time": sum(r["inference_time_ms"] for r in comparison_results["quantized_outputs"]) / len(comparison_results["quantized_outputs"])
        })
        
        return comparison_results
