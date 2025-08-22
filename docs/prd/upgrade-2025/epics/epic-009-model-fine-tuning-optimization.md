# Epic 9: 模型微调和优化平台

**Epic ID**: EPIC-009-MODEL-FINE-TUNING  
**优先级**: 高 (P1)  
**预估工期**: 8-10周  
**负责团队**: AI团队 + MLOps团队  
**创建日期**: 2025-08-19

## 📋 Epic概述

构建完整的模型微调和优化平台，实现LoRA/QLoRA高效微调、模型压缩与量化、自动超参数优化和模型性能评估，让AI Agent系统具备自主模型优化和定制能力，摆脱对外部API的完全依赖。

### 🎯 业务价值
- **模型定制**: 针对特定任务和领域定制优化模型
- **成本优化**: 减少外部API调用，降低运营成本
- **性能提升**: 通过微调获得更好的任务特定性能
- **技术自主**: 掌握完整的模型训练和优化技术栈

## 🚀 核心功能清单

### 1. **LoRA/QLoRA微调框架**
- 基于Hugging Face Transformers的微调流程
- LoRA(Low-Rank Adaptation)高效微调
- QLoRA(Quantized LoRA)量化微调
- 多GPU分布式训练支持

### 2. **模型压缩和量化**
- INT8/INT4量化技术
- 知识蒸馏(Knowledge Distillation)
- 模型剪枝(Pruning)
- 动态量化和静态量化

### 3. **自动超参数优化**
- Optuna/Ray Tune超参数搜索
- 贝叶斯优化和遗传算法
- 早停和学习率调度
- 自动化实验管理

### 4. **模型性能评估系统**
- 多维度评估指标(准确率、延迟、内存)
- 基准测试套件
- A/B测试框架集成
- 性能回归检测

### 5. **训练数据管理**
- 数据收集和标注工具
- 数据质量评估和清洗
- 数据版本控制和管理
- 增量数据集构建

### 6. **模型部署和服务**
- 模型版本管理和回滚
- 多模型并行服务
- 在线学习和增量更新
- 边缘设备部署优化

## 🏗️ 用户故事分解

### Story 9.1: LoRA/QLoRA微调框架
**优先级**: P1 | **工期**: 3周
- 集成Hugging Face PEFT库
- 实现LoRA和QLoRA训练流程
- 支持主流模型架构(LLaMA、Mistral、Qwen等)
- 实现多GPU分布式训练

### Story 9.2: 模型压缩和量化工具
**优先级**: P1 | **工期**: 2-3周
- 实现INT8/INT4量化算法
- 集成知识蒸馏框架
- 实现模型剪枝技术
- 构建压缩效果评估工具

### Story 9.3: 自动超参数优化系统
**优先级**: P1 | **工期**: 2周
- 集成Optuna超参数搜索
- 实现多种优化算法(贝叶斯、遗传、随机搜索)
- 构建实验管理和可视化界面
- 实现早停和资源管理

### Story 9.4: 模型评估和基准测试
**优先级**: P1 | **工期**: 2周
- 构建标准化评估流程
- 实现多维度性能指标
- 集成领域特定基准测试
- 创建性能对比和分析工具

### Story 9.5: 训练数据管理系统
**优先级**: P2 | **工期**: 2周
- 实现数据收集和预处理管道
- 构建数据标注和验证工具
- 实现数据版本控制
- 创建数据质量监控系统

### Story 9.6: 模型服务和部署平台
**优先级**: P1 | **工期**: 2-3周
- 实现模型版本管理
- 构建模型服务API
- 实现在线学习框架
- 集成边缘部署优化

### Story 9.7: 平台集成和优化
**优先级**: P1 | **工期**: 1-2周
- 端到端平台集成
- 性能优化和调试
- 监控告警系统集成
- 文档和培训材料

## 🎯 成功标准 (Definition of Done)

### 技术指标
- ✅ **微调效果**: 任务特定模型相比基础模型提升15%+
- ✅ **压缩比**: 模型大小压缩70%+，性能损失<5%
- ✅ **训练效率**: LoRA微调相比全量微调提升10倍速度
- ✅ **超参数优化**: 自动搜索相比手动调参提升20%+效果
- ✅ **部署延迟**: 量化模型推理延迟<100ms

### 功能指标
- ✅ **模型支持**: 支持10种以上主流开源模型
- ✅ **任务类型**: 支持分类、生成、问答等5种以上任务类型
- ✅ **硬件适配**: 支持CPU、GPU、移动端等多种部署环境
- ✅ **并发能力**: 支持100+并发训练任务
- ✅ **存储管理**: 支持TB级模型和数据存储管理

### 质量标准
- ✅ **测试覆盖率≥90%**: 单元测试 + 集成测试 + E2E测试
- ✅ **训练稳定性**: 95%训练任务成功完成
- ✅ **系统可用性**: 99.5%平台可用性
- ✅ **实验可重现**: 100%实验结果可重现

## 🔧 技术实现亮点

### LoRA/QLoRA微调框架
```python
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FineTuningConfig:
    model_name: str
    task_type: str
    output_dir: str
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # QLoRA配置
    use_quantization: bool = False
    quantization_type: str = "nf4"  # nf4, fp4
    compute_dtype: str = "bfloat16"
    
    # 训练配置
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512
    
    # 分布式训练
    use_multi_gpu: bool = False
    data_parallel: bool = True

class LoRAFineTuner:
    """LoRA/QLoRA微调器"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def prepare_model_and_tokenizer(self):
        """准备模型和分词器"""
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="right",
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量化配置
        quantization_config = None
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.config.use_multi_gpu else None,
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.config.compute_dtype) if not self.config.use_quantization else None
        )
        
        # 为量化训练准备模型
        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True
            )
        
        # 配置LoRA
        if not self.config.target_modules:
            # 根据模型类型自动选择目标模块
            self.config.target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self._print_trainable_parameters()
        
    def _get_target_modules(self) -> List[str]:
        """根据模型架构自动选择LoRA目标模块"""
        model_type = self.model.config.model_type.lower()
        
        target_modules_mapping = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["c_attn", "c_proj", "w1", "w2"],
            "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "baichuan": ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        
        return target_modules_mapping.get(model_type, ["q_proj", "v_proj"])
    
    def _print_trainable_parameters(self):
        """打印可训练参数统计"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        self.logger.info(
            f"Trainable params: {trainable_params} || "
            f"All params: {all_param} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def prepare_dataset(self, train_data: List[Dict], eval_data: Optional[List[Dict]] = None):
        """准备训练数据集"""
        
        def tokenize_function(examples):
            # 构建输入文本
            inputs = []
            for example in examples:
                if self.config.task_type == "instruction_following":
                    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
                elif self.config.task_type == "chat":
                    text = f"Human: {example['input']}\nAssistant: {example['output']}"
                else:
                    text = example.get('text', '')
                
                inputs.append(text)
            
            # 分词
            tokenized = self.tokenizer(
                inputs,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 为因果语言模型设置标签
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 创建数据集
        from datasets import Dataset
        
        train_dataset = Dataset.from_list(train_data)
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = Dataset.from_list(eval_data)
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset, eval_dataset=None):
        """开始训练"""
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            report_to=None,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=not self.config.use_quantization,
            bf16=self.config.use_quantization,
            dataloader_drop_last=False,
            optim="paged_adamw_8bit" if self.config.use_quantization else "adamw_torch",
            remove_unused_columns=False,
        )
        
        # 数据整理器
        from transformers import DataCollatorForLanguageModeling
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        self.logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        
        # 保存训练日志
        with open(f"{self.config.output_dir}/training_log.json", "w") as f:
            json.dump({
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
                "total_flos": train_result.metrics.get("total_flos"),
                "train_loss": train_result.metrics.get("train_loss"),
                "config": self.config.__dict__,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        self.logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        return train_result
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        if not self.trainer:
            raise ValueError("Model not trained yet. Call train() first.")
        
        eval_results = self.trainer.evaluate(eval_dataset)
        
        self.logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def save_adapter(self, save_path: str):
        """保存LoRA适配器"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"LoRA adapter saved to {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """加载LoRA适配器"""
        from peft import PeftModel
        
        # 重新加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载适配器
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        self.logger.info(f"LoRA adapter loaded from {adapter_path}")

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def quantize_dynamic(self, model_path: str, output_path: str, dtype: str = "qint8"):
        """动态量化"""
        import torch.quantization as quantization
        
        # 加载模型
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # 动态量化
        if dtype == "qint8":
            quantized_model = quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        else:
            raise ValueError(f"Unsupported quantization dtype: {dtype}")
        
        # 保存量化模型
        torch.save(quantized_model, output_path)
        
        # 计算压缩比
        original_size = self._get_model_size(model_path)
        quantized_size = self._get_model_size(output_path)
        compression_ratio = original_size / quantized_size
        
        self.logger.info(f"Dynamic quantization completed:")
        self.logger.info(f"Original size: {original_size:.2f} MB")
        self.logger.info(f"Quantized size: {quantized_size:.2f} MB") 
        self.logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio
        }
    
    def quantize_static(self, model, calibration_dataset, output_path: str):
        """静态量化"""
        import torch.quantization as quantization
        
        # 配置量化
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # 校准
        model.eval()
        with torch.no_grad():
            for data in calibration_dataset:
                model(data)
        
        # 转换为量化模型
        quantized_model = quantization.convert(model, inplace=False)
        
        # 保存模型
        torch.save(quantized_model, output_path)
        
        self.logger.info(f"Static quantization completed and saved to {output_path}")
        
        return quantized_model
    
    def _get_model_size(self, model_path: str) -> float:
        """获取模型文件大小(MB)"""
        import os
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, study_name: str = "hyperparameter_optimization"):
        import optuna
        
        self.study_name = study_name
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize(
        self, 
        objective_function, 
        n_trials: int = 100,
        timeout: Optional[int] = None
    ):
        """执行超参数优化"""
        
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # 开始优化
        self.study.optimize(
            objective_function,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 输出最佳结果
        best_trial = self.study.best_trial
        self.logger.info(f"Best trial:")
        self.logger.info(f"  Value: {best_trial.value}")
        self.logger.info(f"  Parameters: {best_trial.params}")
        
        return {
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "n_trials": len(self.study.trials)
        }
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """建议超参数"""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "lora_r": trial.suggest_int("lora_r", 8, 64, step=8),
            "lora_alpha": trial.suggest_int("lora_alpha", 16, 128, step=16),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.05, 0.3),
            "warmup_steps": trial.suggest_int("warmup_steps", 50, 500, step=50),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])
        }
        
        return params
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """获取优化历史"""
        trials = self.study.trials
        
        history = {
            "trial_numbers": [t.number for t in trials],
            "values": [t.value if t.value is not None else 0 for t in trials],
            "parameters": [t.params for t in trials],
            "states": [t.state.name for t in trials]
        }
        
        return history
    
    def plot_optimization_history(self, save_path: str = None):
        """绘制优化历史"""
        import optuna.visualization as vis
        import plotly
        
        # 创建优化历史图
        fig_history = vis.plot_optimization_history(self.study)
        
        # 创建参数重要性图
        fig_importance = vis.plot_param_importances(self.study)
        
        if save_path:
            fig_history.write_html(f"{save_path}/optimization_history.html")
            fig_importance.write_html(f"{save_path}/param_importance.html")
        
        return fig_history, fig_importance

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 评估指标
        self.metrics = {
            "accuracy": self._calculate_accuracy,
            "perplexity": self._calculate_perplexity,
            "bleu": self._calculate_bleu,
            "rouge": self._calculate_rouge,
            "inference_time": self._measure_inference_time,
            "memory_usage": self._measure_memory_usage
        }
    
    def evaluate_model(
        self, 
        model, 
        tokenizer, 
        test_dataset: List[Dict],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """全面评估模型"""
        
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        results = {}
        
        self.logger.info(f"Evaluating model on {len(test_dataset)} samples")
        
        for metric_name in metrics:
            if metric_name in self.metrics:
                self.logger.info(f"Computing {metric_name}...")
                
                try:
                    metric_value = self.metrics[metric_name](model, tokenizer, test_dataset)
                    results[metric_name] = metric_value
                    self.logger.info(f"{metric_name}: {metric_value:.4f}")
                except Exception as e:
                    self.logger.error(f"Error computing {metric_name}: {e}")
                    results[metric_name] = None
        
        return results
    
    def _calculate_accuracy(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """计算准确率"""
        correct = 0
        total = len(test_dataset)
        
        model.eval()
        
        with torch.no_grad():
            for example in test_dataset:
                input_text = example.get('input', '')
                expected_output = example.get('output', '')
                
                # 生成预测
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=100,
                        do_sample=False,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 解码预测结果
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_output = generated_text[len(input_text):].strip()
                
                # 简单匹配检查
                if self._text_similarity(predicted_output, expected_output) > 0.8:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_perplexity(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0
        
        model.eval()
        
        with torch.no_grad():
            for example in test_dataset:
                text = example.get('text', example.get('output', ''))
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
                
                total_loss += loss * inputs.input_ids.size(1)
                total_tokens += inputs.input_ids.size(1)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def _calculate_bleu(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """计算BLEU分数"""
        try:
            from sacrebleu import corpus_bleu
            
            predictions = []
            references = []
            
            model.eval()
            
            for example in test_dataset:
                input_text = example.get('input', '')
                reference = example.get('output', '')
                
                # 生成预测
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated_text[len(input_text):].strip()
                
                predictions.append(prediction)
                references.append(reference)
            
            bleu_score = corpus_bleu(predictions, [references]).score / 100.0
            return bleu_score
            
        except ImportError:
            self.logger.warning("sacrebleu not installed, skipping BLEU calculation")
            return 0.0
    
    def _calculate_rouge(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """计算ROUGE分数"""
        try:
            from rouge import Rouge
            
            rouge_calculator = Rouge()
            predictions = []
            references = []
            
            model.eval()
            
            for example in test_dataset:
                input_text = example.get('input', '')
                reference = example.get('output', '')
                
                # 生成预测
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated_text[len(input_text):].strip()
                
                if prediction and reference:
                    predictions.append(prediction)
                    references.append(reference)
            
            if predictions and references:
                scores = rouge_calculator.get_scores(predictions, references, avg=True)
                return scores['rouge-l']['f']
            else:
                return 0.0
                
        except ImportError:
            self.logger.warning("rouge not installed, skipping ROUGE calculation")
            return 0.0
    
    def _measure_inference_time(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """测量推理时间"""
        import time
        
        model.eval()
        
        total_time = 0
        num_samples = min(50, len(test_dataset))  # 限制测试样本数
        
        for i in range(num_samples):
            example = test_dataset[i]
            input_text = example.get('input', example.get('text', ''))
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_inference_time = total_time / num_samples if num_samples > 0 else 0
        return avg_inference_time * 1000  # 转换为毫秒
    
    def _measure_memory_usage(self, model, tokenizer, test_dataset: List[Dict]) -> float:
        """测量内存使用量"""
        import psutil
        import gc
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 测量基础内存
        process = psutil.Process()
        base_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model.eval()
        
        # 执行推理
        sample = test_dataset[0] if test_dataset else {'input': 'test'}
        input_text = sample.get('input', sample.get('text', 'test'))
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 测量推理后内存
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return peak_memory - base_memory
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        from difflib import SequenceMatcher
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
```

### 模型服务和部署平台
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import asyncio
import time
import json
from datetime import datetime
import logging

class ModelService:
    """模型服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}  # 模型缓存
        self.model_stats = {}  # 模型统计
        self.logger = logging.getLogger(__name__)
        
        # 初始化FastAPI应用
        self.app = FastAPI(title="Model Fine-tuning Service")
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/models/load")
        async def load_model(request: LoadModelRequest):
            """加载模型"""
            try:
                model_id = await self.load_model_async(
                    request.model_name,
                    request.adapter_path,
                    request.device
                )
                return {"model_id": model_id, "status": "loaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_id}/generate")
        async def generate_text(model_id: str, request: GenerationRequest):
            """文本生成"""
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail="Model not found")
            
            try:
                result = await self.generate_async(model_id, request)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}/stats")
        async def get_model_stats(model_id: str):
            """获取模型统计"""
            if model_id not in self.model_stats:
                raise HTTPException(status_code=404, detail="Model not found")
            
            return self.model_stats[model_id]
        
        @self.app.post("/models/{model_id}/unload")
        async def unload_model(model_id: str):
            """卸载模型"""
            try:
                await self.unload_model_async(model_id)
                return {"status": "unloaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            """列出所有已加载的模型"""
            return {
                "models": list(self.models.keys()),
                "total": len(self.models)
            }
    
    async def load_model_async(
        self, 
        model_name: str, 
        adapter_path: Optional[str] = None,
        device: str = "auto"
    ) -> str:
        """异步加载模型"""
        
        model_id = f"{model_name}_{int(time.time())}"
        
        self.logger.info(f"Loading model {model_name} as {model_id}")
        
        # 加载基础模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device != "auto" else "auto"
        )
        
        # 如果有适配器，加载LoRA适配器
        if adapter_path:
            self.logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # 缓存模型
        self.models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": model_name,
            "adapter_path": adapter_path,
            "loaded_at": datetime.now(),
            "device": device
        }
        
        # 初始化统计
        self.model_stats[model_id] = {
            "requests": 0,
            "total_tokens": 0,
            "total_time": 0,
            "avg_latency": 0,
            "last_used": datetime.now()
        }
        
        self.logger.info(f"Model {model_id} loaded successfully")
        return model_id
    
    async def generate_async(self, model_id: str, request: GenerationRequest) -> Dict[str, Any]:
        """异步文本生成"""
        
        start_time = time.time()
        
        model_info = self.models[model_id]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # 编码输入
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=request.max_input_length
        )
        
        # 生成文本
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(request.prompt):].strip()
        
        end_time = time.time()
        
        # 更新统计
        stats = self.model_stats[model_id]
        stats["requests"] += 1
        stats["total_tokens"] += outputs[0].size(0)
        stats["total_time"] += (end_time - start_time)
        stats["avg_latency"] = stats["total_time"] / stats["requests"]
        stats["last_used"] = datetime.now()
        
        return {
            "generated_text": response_text,
            "input_tokens": inputs.input_ids.size(1),
            "output_tokens": outputs[0].size(0) - inputs.input_ids.size(1),
            "latency_ms": (end_time - start_time) * 1000,
            "model_id": model_id
        }
    
    async def unload_model_async(self, model_id: str):
        """异步卸载模型"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.logger.info(f"Unloading model {model_id}")
        
        # 清理内存
        del self.models[model_id]
        del self.model_stats[model_id]
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Model {model_id} unloaded successfully")

# Pydantic模型
class LoadModelRequest(BaseModel):
    model_name: str
    adapter_path: Optional[str] = None
    device: str = "auto"

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    max_input_length: int = 512

class ModelManager:
    """模型管理器"""
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = storage_path
        self.model_registry = {}
        self.version_history = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(
        self, 
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any]
    ):
        """注册模型版本"""
        
        model_key = f"{model_name}:{version}"
        
        self.model_registry[model_key] = {
            "model_name": model_name,
            "version": version,
            "model_path": model_path,
            "metadata": metadata,
            "registered_at": datetime.now(),
            "status": "active"
        }
        
        # 更新版本历史
        if model_name not in self.version_history:
            self.version_history[model_name] = []
        
        self.version_history[model_name].append({
            "version": version,
            "registered_at": datetime.now(),
            "metadata": metadata
        })
        
        self.logger.info(f"Model {model_key} registered successfully")
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """获取模型信息"""
        
        if version == "latest":
            # 获取最新版本
            if model_name in self.version_history:
                latest_version = max(
                    self.version_history[model_name],
                    key=lambda x: x["registered_at"]
                )["version"]
                model_key = f"{model_name}:{latest_version}"
            else:
                return None
        else:
            model_key = f"{model_name}:{version}"
        
        return self.model_registry.get(model_key)
    
    def list_models(self) -> Dict[str, List[str]]:
        """列出所有模型"""
        models = {}
        
        for model_key in self.model_registry:
            model_name, version = model_key.split(":", 1)
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(version)
        
        return models
    
    def rollback_model(self, model_name: str, target_version: str):
        """回滚模型版本"""
        
        current_info = self.get_model_info(model_name, "latest")
        target_info = self.get_model_info(model_name, target_version)
        
        if not target_info:
            raise ValueError(f"Target version {target_version} not found")
        
        # 标记当前版本为非激活
        if current_info:
            current_key = f"{model_name}:{current_info['version']}"
            self.model_registry[current_key]["status"] = "inactive"
        
        # 激活目标版本
        target_key = f"{model_name}:{target_version}"
        self.model_registry[target_key]["status"] = "active"
        
        self.logger.info(f"Model {model_name} rolled back to version {target_version}")
    
    def cleanup_old_versions(self, model_name: str, keep_versions: int = 5):
        """清理旧版本"""
        
        if model_name not in self.version_history:
            return
        
        versions = sorted(
            self.version_history[model_name],
            key=lambda x: x["registered_at"],
            reverse=True
        )
        
        if len(versions) <= keep_versions:
            return
        
        # 删除多余的版本
        for version_info in versions[keep_versions:]:
            version = version_info["version"]
            model_key = f"{model_name}:{version}"
            
            if model_key in self.model_registry:
                model_path = self.model_registry[model_key]["model_path"]
                
                # 删除模型文件
                import shutil
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                
                # 从注册表中删除
                del self.model_registry[model_key]
        
        # 更新版本历史
        self.version_history[model_name] = versions[:keep_versions]
        
        self.logger.info(f"Cleaned up old versions for model {model_name}")
```

## 🚦 风险评估与缓解

### 高风险项
1. **训练资源需求高**
   - 缓解: 云端训练资源管理，成本优化算法
   - 验证: 资源使用监控和预警系统

2. **模型微调效果不稳定**
   - 缓解: 多种微调策略组合，自动超参数优化
   - 验证: 大量实验验证和效果基准测试

3. **量化后性能损失**
   - 缓解: 渐进式量化，质量监控和回退机制
   - 验证: 量化前后性能对比测试

### 中风险项
1. **模型部署复杂性**
   - 缓解: 容器化部署，自动化CI/CD流程
   - 验证: 部署稳定性和回滚测试

2. **训练数据质量**
   - 缓解: 数据清洗和验证流程，质量评估指标
   - 验证: 数据质量审核和标注一致性检查

## 📅 实施路线图

### Phase 1: 基础框架搭建 (Week 1-3)
- LoRA/QLoRA微调框架
- 模型压缩和量化工具
- 基础训练流程

### Phase 2: 优化和评估 (Week 4-6)
- 自动超参数优化系统
- 模型评估和基准测试
- 性能监控集成

### Phase 3: 数据和服务 (Week 7-8)
- 训练数据管理系统
- 模型服务和部署平台
- API接口和管理工具

### Phase 4: 集成优化 (Week 9-10)
- 端到端平台集成
- 性能优化调试
- 生产环境部署

---

**文档状态**: ✅ 完成  
**下一步**: 开始Story 9.1的LoRA/QLoRA微调框架实施  
**依赖Epic**: 无强依赖，可独立开发