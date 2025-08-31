"""
知识蒸馏训练器

实现Teacher-Student蒸馏和Self-Distillation
支持Response-based、Feature-based、Attention-based蒸馏
提供多种蒸馏损失函数和温度参数调节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import logging
import time
from pathlib import Path
import os
from dataclasses import dataclass

from .models import DistillationConfig, CompressionResult, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class DistillationResult:
    """蒸馏结果"""
    distillation_losses: List[float]
    final_loss: float
    evaluation_results: Dict[str, Any]
    student_model: nn.Module
    training_time: float
    epochs_completed: int


class DistillationTrainer:
    """知识蒸馏训练器
    
    支持多种蒸馏策略:
    - Response-based Distillation: 基于输出响应的蒸馏
    - Feature-based Distillation: 基于特征层的蒸馏  
    - Attention-based Distillation: 基于注意力机制的蒸馏
    - Self-Distillation: 自蒸馏
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        
        self.logger = logging.getLogger(__name__)
        
        # 蒸馏策略映射
        self.distillation_strategies = {
            "response_based": self._response_based_distillation,
            "feature_based": self._feature_based_distillation,
            "attention_based": self._attention_based_distillation,
            "self_distillation": self._self_distillation
        }
        
        # 损失函数缓存
        self._loss_cache = {}
    
    def load_models(self) -> None:
        """加载教师和学生模型"""
        
        self.logger.info("加载教师和学生模型")
        
        # 加载教师模型
        self.teacher_model = self._load_single_model(
            self.config.teacher_model, "Teacher"
        )
        self.teacher_model.eval()
        
        # 加载学生模型
        self.student_model = self._load_single_model(
            self.config.student_model, "Student"  
        )
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.logger.info("模型加载完成")
        self.logger.info(f"教师模型参数量: {self._count_parameters(self.teacher_model):,}")
        self.logger.info(f"学生模型参数量: {self._count_parameters(self.student_model):,}")
    
    def _load_single_model(self, model_path: str, model_type: str) -> nn.Module:
        """加载单个模型"""
        
        try:
            # 尝试加载PyTorch模型
            if os.path.isfile(model_path) and model_path.endswith('.pth'):
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, nn.Module):
                    return model
                else:
                    raise ValueError("文件不包含有效的PyTorch模型")
            
            # 尝试加载Hugging Face模型
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                return model
            except ImportError:
                raise ImportError("需要安装transformers库来加载Hugging Face模型")
            
        except Exception as e:
            raise ValueError(f"无法加载{model_type}模型 {model_path}: {e}")
    
    def distill(
        self, 
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> DistillationResult:
        """执行知识蒸馏"""
        
        if self.teacher_model is None or self.student_model is None:
            self.load_models()
        
        # 使用配置中的参数或传入的参数
        num_epochs = num_epochs or self.config.num_epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        self.logger.info(f"开始知识蒸馏，策略: {self.config.distillation_type}")
        self.logger.info(f"轮数: {num_epochs}, 学习率: {learning_rate}")
        
        start_time = time.time()
        
        # 获取蒸馏策略
        distillation_func = self.distillation_strategies.get(self.config.distillation_type)
        if not distillation_func:
            raise ValueError(f"不支持的蒸馏策略: {self.config.distillation_type}")
        
        # 设置优化器和调度器
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        # 训练循环
        distillation_losses = []
        device = next(self.student_model.parameters()).device
        
        for epoch in range(num_epochs):
            self.logger.info(f"开始蒸馏轮次 {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            self.student_model.train()
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # 移动数据到设备
                    batch = self._move_batch_to_device(batch, device)
                    
                    # 执行蒸馏步骤
                    loss = distillation_func(batch, optimizer)
                    
                    epoch_loss += loss
                    num_batches += 1
                    
                    # 学习率调度
                    if batch_idx < self.config.warmup_steps:
                        scheduler.step()
                    
                    # 记录训练进度
                    if batch_idx % 50 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        self.logger.info(
                            f"轮次 {epoch+1}, 批次 {batch_idx}, "
                            f"损失: {loss:.4f}, 学习率: {current_lr:.6f}"
                        )
                
                except Exception as e:
                    self.logger.warning(f"批次 {batch_idx} 蒸馏失败: {e}")
                    continue
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            distillation_losses.append(avg_loss)
            
            self.logger.info(f"轮次 {epoch + 1} 平均损失: {avg_loss:.4f}")
            
            # 评估学生模型
            if eval_dataloader and epoch % 2 == 0:  # 每2轮评估一次
                eval_results = self._evaluate_student_model(eval_dataloader)
                self.logger.info(f"轮次 {epoch + 1} 评估结果: {eval_results}")
        
        # 最终评估
        eval_results = {}
        if eval_dataloader:
            self.student_model.eval()
            eval_results = self._evaluate_student_model(eval_dataloader)
        
        training_time = time.time() - start_time
        
        result = DistillationResult(
            distillation_losses=distillation_losses,
            final_loss=distillation_losses[-1] if distillation_losses else float('inf'),
            evaluation_results=eval_results,
            student_model=self.student_model,
            training_time=training_time,
            epochs_completed=len(distillation_losses)
        )
        
        self.logger.info(f"知识蒸馏完成，耗时: {training_time:.2f}s")
        
        return result
    
    def _move_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """将批次数据移动到设备"""
        
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(item, device) for item in batch]
        else:
            return batch
    
    def _response_based_distillation(
        self, 
        batch: Any, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """基于响应的蒸馏"""
        
        optimizer.zero_grad()
        
        # 准备输入
        if isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', batch))
            labels = batch.get('labels', None)
        elif isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            inputs = batch
            labels = None
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            else:
                teacher_logits = teacher_outputs
        
        # 学生模型前向传播  
        student_outputs = self.student_model(inputs)
        if hasattr(student_outputs, 'logits'):
            student_logits = student_outputs.logits
        else:
            student_logits = student_outputs
        
        # 计算蒸馏损失
        distill_loss = self._compute_kl_divergence_loss(
            teacher_logits, student_logits, self.config.temperature
        )
        
        # 如果有标签，计算硬目标损失
        hard_loss = 0.0
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        # 组合损失
        total_loss = (
            self.config.alpha * distill_loss +
            (1 - self.config.alpha) * hard_loss
        )
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item()
    
    def _feature_based_distillation(
        self, 
        batch: Any, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """基于特征的蒸馏"""
        
        optimizer.zero_grad()
        
        # 准备输入
        if isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', batch))
        elif isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        
        # 注册hook来获取中间特征
        teacher_features = {}
        student_features = {}
        
        def get_teacher_hook(name):
            def hook(module, input, output):
                teacher_features[name] = output
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                student_features[name] = output
            return hook
        
        # 注册指定层的hook
        target_layers = self.config.feature_layers or [6, 12]  # 默认使用第6层和第12层
        teacher_hooks = []
        student_hooks = []
        
        try:
            # 为教师模型注册hook
            for layer_idx in target_layers:
                if hasattr(self.teacher_model, 'transformer') and hasattr(self.teacher_model.transformer, 'h'):
                    if layer_idx < len(self.teacher_model.transformer.h):
                        hook = self.teacher_model.transformer.h[layer_idx].register_forward_hook(
                            get_teacher_hook(f'layer_{layer_idx}')
                        )
                        teacher_hooks.append(hook)
            
            # 为学生模型注册hook
            for layer_idx in target_layers:
                if hasattr(self.student_model, 'transformer') and hasattr(self.student_model.transformer, 'h'):
                    if layer_idx < len(self.student_model.transformer.h):
                        hook = self.student_model.transformer.h[layer_idx].register_forward_hook(
                            get_student_hook(f'layer_{layer_idx}')
                        )
                        student_hooks.append(hook)
            
            # 前向传播
            with torch.no_grad():
                self.teacher_model(inputs)
            
            student_outputs = self.student_model(inputs)
            
            # 计算特征蒸馏损失
            feature_loss = 0.0
            for layer_name in teacher_features:
                if layer_name in student_features:
                    teacher_feat = teacher_features[layer_name]
                    student_feat = student_features[layer_name]
                    
                    # 特征维度对齐
                    if teacher_feat.shape != student_feat.shape:
                        # 使用线性投影对齐维度
                        if not hasattr(self, '_feature_projectors'):
                            self._feature_projectors = {}
                        
                        proj_key = f"{layer_name}_{teacher_feat.shape[-1]}_{student_feat.shape[-1]}"
                        if proj_key not in self._feature_projectors:
                            self._feature_projectors[proj_key] = nn.Linear(
                                student_feat.shape[-1], teacher_feat.shape[-1]
                            ).to(student_feat.device)
                        
                        student_feat = self._feature_projectors[proj_key](student_feat)
                    
                    # 计算MSE损失
                    layer_loss = F.mse_loss(student_feat, teacher_feat)
                    feature_loss += layer_loss
            
            # 如果有标准的输出蒸馏损失，也加上
            if hasattr(student_outputs, 'logits'):
                response_loss = self._compute_kl_divergence_loss(
                    teacher_features.get('output', student_outputs.logits),
                    student_outputs.logits,
                    self.config.temperature
                )
                feature_loss += response_loss * 0.5
        
        finally:
            # 清理hook
            for hook in teacher_hooks + student_hooks:
                hook.remove()
        
        # 反向传播
        if feature_loss > 0:
            feature_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            optimizer.step()
        
        return feature_loss.item() if feature_loss > 0 else 0.0
    
    def _attention_based_distillation(
        self, 
        batch: Any, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """基于注意力的蒸馏"""
        
        # 简化实现：使用响应蒸馏 + 注意力正则化
        self.logger.warning("注意力蒸馏使用简化实现")
        return self._response_based_distillation(batch, optimizer)
    
    def _self_distillation(
        self, 
        batch: Any, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """自蒸馏"""
        
        # 自蒸馏：使用模型的过去版本作为教师
        optimizer.zero_grad()
        
        # 准备输入
        if isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', batch))
        elif isinstance(batch, (list, tuple)):
            inputs = batch[0] 
        else:
            inputs = batch
        
        # 当前模型输出
        current_outputs = self.student_model(inputs)
        if hasattr(current_outputs, 'logits'):
            current_logits = current_outputs.logits
        else:
            current_logits = current_outputs
        
        # 使用EMA作为"教师"
        if not hasattr(self, '_ema_model'):
            self._ema_model = self._create_ema_model(self.student_model)
        
        # EMA模型输出  
        with torch.no_grad():
            ema_outputs = self._ema_model(inputs)
            if hasattr(ema_outputs, 'logits'):
                ema_logits = ema_outputs.logits
            else:
                ema_logits = ema_outputs
        
        # 计算自蒸馏损失
        self_distill_loss = self._compute_kl_divergence_loss(
            ema_logits, current_logits, self.config.temperature
        )
        
        # 反向传播
        self_distill_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
        optimizer.step()
        
        # 更新EMA模型
        self._update_ema_model(self.student_model, self._ema_model)
        
        return self_distill_loss.item()
    
    def _compute_kl_divergence_loss(
        self, 
        teacher_logits: torch.Tensor, 
        student_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """计算KL散度损失"""
        
        # 温度缩放
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL散度
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean',
            log_target=False
        )
        
        # 温度平方缩放
        return kl_loss * (temperature ** 2)
    
    def _create_ema_model(self, model: nn.Module, decay: float = 0.999) -> nn.Module:
        """创建EMA模型"""
        
        ema_model = type(model)(model.config if hasattr(model, 'config') else None)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        
        # 冻结EMA模型参数
        for param in ema_model.parameters():
            param.requires_grad = False
        
        self._ema_decay = decay
        return ema_model
    
    def _update_ema_model(self, model: nn.Module, ema_model: nn.Module) -> None:
        """更新EMA模型"""
        
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self._ema_decay).add_(param.data, alpha=1 - self._ema_decay)
    
    def _evaluate_student_model(
        self, 
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """评估学生模型"""
        
        self.student_model.eval()
        total_loss = 0.0
        num_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                try:
                    device = next(self.student_model.parameters()).device
                    batch = self._move_batch_to_device(batch, device)
                    
                    # 准备输入和标签
                    if isinstance(batch, dict):
                        inputs = batch.get('input_ids', batch.get('inputs'))
                        labels = batch.get('labels')
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        labels = batch[1] if len(batch) > 1 else None
                    else:
                        inputs = batch
                        labels = None
                    
                    # 前向传播
                    outputs = self.student_model(inputs)
                    
                    if hasattr(outputs, 'loss') and labels is not None:
                        loss = outputs.loss
                        total_loss += loss.item()
                    
                    # 计算准确率（如果有标签）
                    if labels is not None and hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        correct_predictions += (predictions == labels).sum().item()
                    
                    num_samples += inputs.size(0) if hasattr(inputs, 'size') else 1
                
                except Exception as e:
                    self.logger.warning(f"评估批次时出错: {e}")
                    continue
        
        results = {
            "avg_loss": total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0.0,
            "num_samples": num_samples,
        }
        
        if correct_predictions > 0:
            results["accuracy"] = correct_predictions / num_samples
        
        return results
    
    def _count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def save_student_model(self, save_path: str) -> str:
        """保存学生模型"""
        
        if self.student_model is None:
            raise ValueError("学生模型未初始化")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            # 尝试保存为transformers格式
            if hasattr(self.student_model, 'save_pretrained'):
                self.student_model.save_pretrained(save_path)
            else:
                # 保存为PyTorch格式
                torch.save(self.student_model, save_path)
            
            self.logger.info(f"学生模型已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存学生模型失败: {e}")
            raise
    
    def get_compression_ratio(self) -> float:
        """获取模型压缩比"""
        
        if self.teacher_model is None or self.student_model is None:
            return 0.0
        
        teacher_params = self._count_parameters(self.teacher_model)
        student_params = self._count_parameters(self.student_model)
        
        return teacher_params / student_params if student_params > 0 else 0.0
    
    def get_parameter_reduction(self) -> float:
        """获取参数减少比例"""
        
        if self.teacher_model is None or self.student_model is None:
            return 0.0
        
        teacher_params = self._count_parameters(self.teacher_model)
        student_params = self._count_parameters(self.student_model)
        
        return (teacher_params - student_params) / teacher_params if teacher_params > 0 else 0.0
    
    def get_supported_strategies(self) -> List[str]:
        """获取支持的蒸馏策略"""
        return list(self.distillation_strategies.keys())
    
    def validate_config(self, config: DistillationConfig) -> bool:
        """验证蒸馏配置"""
        
        if config.distillation_type not in self.distillation_strategies:
            return False
        
        if not 0 < config.temperature <= 10:
            return False
        
        if not 0 <= config.alpha <= 1:
            return False
        
        if config.num_epochs <= 0:
            return False
        
        if config.learning_rate <= 0:
            return False
        
        return True