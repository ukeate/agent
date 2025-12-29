"""
模型剪枝引擎

实现结构化和非结构化剪枝算法
支持权重剪枝、神经元剪枝、层剪枝
提供渐进式剪枝和一次性剪枝策略
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import time
import copy
from dataclasses import dataclass
from .models import PruningConfig, PruningType, CompressionResult

from src.core.logging import get_logger
@dataclass
class PruningResult:
    """剪枝结果"""
    original_params: int
    pruned_params: int
    param_reduction: float
    original_flops: int
    pruned_flops: int
    flops_reduction: float
    sparsity_ratio: float
    pruning_time: float
    pruning_method: str
    recovery_applied: bool = False
    recovery_epochs: int = 0

class PruningEngine:
    """模型剪枝引擎
    
    支持多种剪枝策略:
    - Unstructured Pruning: 非结构化剪枝（权重级别）
    - Structured Pruning: 结构化剪枝（神经元、通道级别）
    - Magnitude-based Pruning: 基于权重大小的剪枝
    - Gradient-based Pruning: 基于梯度信息的剪枝
    """
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 剪枝策略映射
        self.pruning_strategies = {
            PruningType.UNSTRUCTURED: self._unstructured_pruning,
            PruningType.STRUCTURED: self._structured_pruning,
            PruningType.MAGNITUDE: self._magnitude_based_pruning,
            PruningType.GRADIENT: self._gradient_based_pruning
        }
        
        # 重要性度量函数
        self.importance_metrics = {
            "magnitude": self._magnitude_importance,
            "gradient": self._gradient_importance,
            "fisher": self._fisher_importance,
            "random": self._random_importance
        }
        
        # 剪枝历史
        self.pruning_history = []
    
    def prune_model(
        self, 
        model: nn.Module,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Tuple[nn.Module, PruningResult]:
        """执行模型剪枝"""
        
        start_time = time.time()
        
        self.logger.info(f"开始模型剪枝，方法: {self.config.pruning_type.value}")
        self.logger.info(f"目标稀疏度: {self.config.sparsity_ratio}")
        
        # 获取原始模型信息
        original_params = self._count_parameters(model)
        original_flops = self._estimate_flops(model)
        
        self.logger.info(f"原始模型参数量: {original_params:,}")
        self.logger.info(f"原始模型FLOPs: {original_flops:,}")
        
        # 创建模型副本
        pruned_model = copy.deepcopy(model)
        
        # 获取剪枝策略
        pruning_func = self.pruning_strategies.get(self.config.pruning_type)
        if not pruning_func:
            raise ValueError(f"不支持的剪枝类型: {self.config.pruning_type}")
        
        # 执行剪枝
        if self.config.gradual_pruning:
            pruned_model = self._gradual_pruning(
                pruned_model, pruning_func, train_dataloader, val_dataloader
            )
        else:
            pruned_model = pruning_func(pruned_model, self.config.sparsity_ratio)
        
        # 恢复训练（如果需要）
        recovery_applied = False
        recovery_epochs = 0
        if self.config.gradual_pruning and train_dataloader and self.config.recovery_epochs > 0:
            self.logger.info("开始恢复训练")
            recovery_epochs = self._recovery_finetuning(
                pruned_model, train_dataloader, self.config.recovery_epochs
            )
            recovery_applied = True
        
        # 永久化剪枝
        self._make_pruning_permanent(pruned_model)
        
        # 计算剪枝效果
        pruned_params = self._count_parameters(pruned_model)
        pruned_flops = self._estimate_flops(pruned_model)
        pruning_time = time.time() - start_time
        
        result = PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            param_reduction=(original_params - pruned_params) / original_params,
            original_flops=original_flops,
            pruned_flops=pruned_flops,
            flops_reduction=(original_flops - pruned_flops) / original_flops,
            sparsity_ratio=self.config.sparsity_ratio,
            pruning_time=pruning_time,
            pruning_method=self.config.pruning_type.value,
            recovery_applied=recovery_applied,
            recovery_epochs=recovery_epochs
        )
        
        self.logger.info(f"剪枝完成！参数减少: {result.param_reduction*100:.1f}%")
        self.logger.info(f"FLOPs减少: {result.flops_reduction*100:.1f}%")
        self.logger.info(f"剪枝耗时: {pruning_time:.2f}s")
        
        return pruned_model, result
    
    def _gradual_pruning(
        self,
        model: nn.Module,
        pruning_func: Callable,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_steps: int = 5
    ) -> nn.Module:
        """渐进式剪枝"""
        
        self.logger.info(f"执行渐进式剪枝，分{num_steps}步完成")
        
        target_sparsity = self.config.sparsity_ratio
        current_sparsity = 0.0
        sparsity_step = target_sparsity / num_steps
        
        for step in range(num_steps):
            current_sparsity = min(current_sparsity + sparsity_step, target_sparsity)
            
            self.logger.info(f"剪枝步骤 {step+1}/{num_steps}, 当前稀疏度: {current_sparsity:.3f}")
            
            # 执行当前步的剪枝
            model = pruning_func(model, current_sparsity)
            
            # 如果有训练数据，进行少量恢复训练
            if train_dataloader and step < num_steps - 1:  # 最后一步不需要中间恢复
                recovery_epochs = max(1, self.config.recovery_epochs // num_steps)
                self._recovery_finetuning(model, train_dataloader, recovery_epochs)
            
            # 验证性能
            if val_dataloader:
                val_loss = self._evaluate_model(model, val_dataloader)
                self.logger.info(f"步骤 {step+1} 验证损失: {val_loss:.4f}")
        
        return model
    
    def _unstructured_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """非结构化剪枝"""
        
        self.logger.info(f"执行非结构化剪枝，稀疏度: {sparsity_ratio}")
        
        # 获取所有目标模块
        modules_to_prune = self._get_target_modules(model)
        
        if not modules_to_prune:
            self.logger.warning("未找到可剪枝的模块")
            return model
        
        # 应用全局非结构化剪枝
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_ratio
        )
        
        self.logger.info(f"非结构化剪枝完成，处理了{len(modules_to_prune)}个模块")
        
        return model
    
    def _structured_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """结构化剪枝"""
        
        self.logger.info(f"执行结构化剪枝，稀疏度: {sparsity_ratio}")
        
        # 获取重要性评估函数
        importance_func = self.importance_metrics.get(
            self.config.importance_metric, self._magnitude_importance
        )
        
        # 遍历所有模块进行结构化剪枝
        for name, module in model.named_modules():
            if self._should_prune_module(name, module):
                self._prune_module_structured(module, sparsity_ratio, importance_func)
        
        return model
    
    def _prune_module_structured(
        self, 
        module: nn.Module, 
        sparsity_ratio: float,
        importance_func: Callable
    ) -> None:
        """对单个模块执行结构化剪枝"""
        
        if isinstance(module, nn.Linear):
            self._prune_linear_structured(module, sparsity_ratio, importance_func)
        elif isinstance(module, nn.Conv2d):
            self._prune_conv2d_structured(module, sparsity_ratio, importance_func)
    
    def _prune_linear_structured(
        self, 
        module: nn.Linear, 
        sparsity_ratio: float,
        importance_func: Callable
    ) -> None:
        """结构化剪枝线性层（剪枝神经元）"""
        
        weight = module.weight.data  # shape: [out_features, in_features]
        
        # 计算每个输出神经元的重要性
        importance_scores = importance_func(weight, dim=1)  # 沿in_features维度
        
        # 确定要剪枝的神经元数量
        num_neurons = weight.size(0)
        num_to_prune = int(num_neurons * sparsity_ratio)
        
        if num_to_prune == 0:
            return
        
        # 选择最不重要的神经元
        _, prune_indices = torch.topk(
            importance_scores, num_to_prune, largest=False
        )
        
        # 创建剪枝mask
        mask = torch.ones_like(importance_scores, dtype=torch.bool)
        mask[prune_indices] = False
        
        # 应用结构化剪枝mask
        prune.custom_from_mask(module, 'weight', mask.unsqueeze(1).expand_as(weight))
        
        # 如果有bias，也要剪枝对应的bias
        if module.bias is not None:
            prune.custom_from_mask(module, 'bias', mask)
    
    def _prune_conv2d_structured(
        self, 
        module: nn.Conv2d, 
        sparsity_ratio: float,
        importance_func: Callable
    ) -> None:
        """结构化剪枝卷积层（剪枝通道）"""
        
        weight = module.weight.data  # shape: [out_channels, in_channels, h, w]
        
        # 计算每个输出通道的重要性
        # 对每个输出通道的所有权重求和作为重要性度量
        importance_scores = importance_func(
            weight.view(weight.size(0), -1), dim=1
        )
        
        # 确定要剪枝的通道数量
        num_channels = weight.size(0)
        num_to_prune = int(num_channels * sparsity_ratio)
        
        if num_to_prune == 0:
            return
        
        # 选择最不重要的通道
        _, prune_indices = torch.topk(
            importance_scores, num_to_prune, largest=False
        )
        
        # 创建剪枝mask
        mask = torch.ones(num_channels, dtype=torch.bool, device=weight.device)
        mask[prune_indices] = False
        
        # 应用结构化剪枝mask
        channel_mask = mask.view(-1, 1, 1, 1).expand_as(weight)
        prune.custom_from_mask(module, 'weight', channel_mask)
        
        # 如果有bias，也要剪枝对应的bias
        if module.bias is not None:
            prune.custom_from_mask(module, 'bias', mask)
    
    def _magnitude_based_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """基于权重大小的剪枝"""
        
        self.logger.info(f"执行基于权重大小的剪枝，稀疏度: {sparsity_ratio}")
        
        # 收集所有权重及其重要性分数
        all_weights = []
        module_info = []
        
        for name, module in model.named_modules():
            if self._should_prune_module(name, module):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data.flatten()
                    importance = torch.abs(weight)
                    
                    all_weights.append(weight)
                    module_info.append((name, module, len(weight)))
        
        if not all_weights:
            self.logger.warning("未找到可剪枝的权重")
            return model
        
        # 合并所有权重
        all_weights = torch.cat(all_weights)
        all_importance = torch.abs(all_weights)
        
        # 找到全局阈值
        num_to_prune = int(len(all_weights) * sparsity_ratio)
        threshold_value = torch.topk(all_importance, num_to_prune, largest=False)[0][-1]
        
        # 应用剪枝到各个模块
        weight_idx = 0
        for name, module, weight_len in module_info:
            weight_importance = all_importance[weight_idx:weight_idx + weight_len]
            mask = weight_importance > threshold_value
            
            # 重塑mask到原始权重形状
            mask = mask.view(module.weight.shape)
            prune.custom_from_mask(module, 'weight', mask)
            
            weight_idx += weight_len
        
        self.logger.info(f"基于权重大小的剪枝完成，阈值: {threshold_value:.6f}")
        
        return model
    
    def _gradient_based_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """基于梯度信息的剪枝"""
        
        self.logger.info(f"执行基于梯度的剪枝，稀疏度: {sparsity_ratio}")
        self.logger.warning("梯度剪枝需要训练数据，当前使用权重大小作为替代")
        
        # 在没有梯度信息的情况下，退回到基于权重大小的剪枝
        return self._magnitude_based_pruning(model, sparsity_ratio)
    
    def _magnitude_importance(self, tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
        """基于权重大小计算重要性"""
        if dim is None:
            return torch.abs(tensor)
        else:
            return torch.sum(torch.abs(tensor), dim=dim)
    
    def _gradient_importance(self, tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
        """基于梯度计算重要性"""
        # 简化实现：使用权重大小作为替代
        return self._magnitude_importance(tensor, dim)
    
    def _fisher_importance(self, tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
        """基于Fisher信息计算重要性"""
        # 简化实现：使用权重大小的平方作为替代
        if dim is None:
            return tensor ** 2
        else:
            return torch.sum(tensor ** 2, dim=dim)
    
    def _random_importance(self, tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
        """随机重要性（用于测试）"""
        if dim is None:
            return torch.rand_like(tensor)
        else:
            shape = list(tensor.shape)
            shape[dim] = 1
            return torch.rand(shape).squeeze(dim).to(tensor.device)
    
    def _get_target_modules(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """获取目标剪枝模块"""
        
        modules_to_prune = []
        
        for name, module in model.named_modules():
            if self._should_prune_module(name, module):
                modules_to_prune.append((module, 'weight'))
        
        return modules_to_prune
    
    def _should_prune_module(self, name: str, module: nn.Module) -> bool:
        """判断是否应该剪枝该模块"""
        
        # 基本类型检查
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            return False
        
        # 检查是否有权重
        if not hasattr(module, 'weight') or module.weight is None:
            return False
        
        # 检查目标模块配置
        if self.config.target_modules:
            # 如果指定了目标模块，只处理指定的模块
            return any(target in name for target in self.config.target_modules)
        
        # 默认处理所有支持的模块类型
        return True
    
    def _recovery_finetuning(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        epochs: int
    ) -> int:
        """恢复微调"""
        
        self.logger.info(f"开始恢复微调，轮次: {epochs}")
        
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.recovery_learning_rate,
            weight_decay=0.01
        )
        
        completed_epochs = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    optimizer.zero_grad()
                    
                    # 处理不同的批次格式
                    if isinstance(batch, dict):
                        inputs = batch.get('input_ids', batch.get('inputs'))
                        targets = batch.get('labels', inputs)
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        targets = batch[1] if len(batch) > 1 else inputs
                    else:
                        inputs = targets = batch
                    
                    # 确保数据在正确的设备上
                    device = next(model.parameters()).device
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(device)
                    if hasattr(targets, 'to'):
                        targets = targets.to(device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    
                    # 计算损失
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        # 简化的损失计算
                        if hasattr(outputs, 'logits'):
                            loss = F.cross_entropy(
                                outputs.logits.view(-1, outputs.logits.size(-1)),
                                targets.view(-1),
                                ignore_index=-100
                            )
                        else:
                            loss = F.mse_loss(outputs, targets.float())
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # 限制每个epoch的批次数（避免过度训练）
                    if batch_idx >= 100:  # 最多100个批次
                        break
                
                except Exception as e:
                    self.logger.warning(f"恢复微调批次 {batch_idx} 时出错: {e}")
                    continue
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            completed_epochs += 1
            
            if epoch % max(1, epochs // 5) == 0:  # 每20%进度报告一次
                self.logger.info(f"恢复微调轮次 {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
        
        self.logger.info(f"恢复微调完成，完成轮次: {completed_epochs}")
        return completed_epochs
    
    def _evaluate_model(
        self, 
        model: nn.Module, 
        val_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """评估模型性能"""
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        last_error: Optional[Exception] = None
        
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    device = next(model.parameters()).device
                    if isinstance(batch, dict):
                        inputs = {
                            k: (v.to(device) if hasattr(v, "to") else v)
                            for k, v in batch.items()
                        }
                        outputs = model(**inputs)
                    else:
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0]
                        else:
                            inputs = batch

                        if hasattr(inputs, "to"):
                            inputs = inputs.to(device)
                        outputs = model(inputs)
                    
                    loss = getattr(outputs, "loss", None)
                    if loss is None:
                        raise ValueError("模型评估需要loss输出（请提供labels）")
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 限制评估批次
                    if num_batches >= 50:
                        break
                
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"评估批次时出错: {e}")
                    continue

        if num_batches <= 0:
            raise ValueError(f"模型评估失败: {last_error}")

        return total_loss / num_batches
    
    def _make_pruning_permanent(self, model: nn.Module) -> None:
        """永久化剪枝（移除mask，直接修改权重）"""
        
        self.logger.info("永久化剪枝mask")
        
        for name, module in model.named_modules():
            if prune.is_pruned(module):
                # 获取所有被剪枝的参数
                pruned_params = []
                for param_name, _ in module.named_parameters():
                    if param_name.endswith('_orig'):
                        base_name = param_name[:-5]  # 移除'_orig'后缀
                        pruned_params.append(base_name)
                
                # 永久化每个被剪枝的参数
                for param_name in pruned_params:
                    try:
                        prune.remove(module, param_name)
                    except Exception as e:
                        self.logger.warning(f"永久化剪枝参数 {name}.{param_name} 时出错: {e}")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _estimate_flops(self, model: nn.Module) -> int:
        """估算模型FLOPs（简化实现）"""
        
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # 线性层：input_features × output_features
                if hasattr(module, 'weight') and module.weight is not None:
                    total_flops += module.weight.numel()
            
            elif isinstance(module, nn.Conv2d):
                # 卷积层：更复杂的计算，这里简化
                if hasattr(module, 'weight') and module.weight is not None:
                    # 权重数量 × 输出特征图大小的估算
                    total_flops += module.weight.numel() * 64  # 假设输出特征图为64x64
        
        return total_flops
    
    def get_sparsity_stats(self, model: nn.Module) -> Dict[str, float]:
        """获取模型稀疏度统计"""
        
        total_params = 0
        zero_params = 0
        layer_stats = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                layer_total = weight.numel()
                layer_zero = (weight == 0).sum().item()
                
                total_params += layer_total
                zero_params += layer_zero
                
                if layer_total > 0:
                    layer_sparsity = layer_zero / layer_total
                    layer_stats[name] = layer_sparsity
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            "overall_sparsity": overall_sparsity,
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "layer_statistics": layer_stats
        }
    
    def validate_config(self, config: PruningConfig) -> bool:
        """验证剪枝配置"""
        
        if config.pruning_type not in self.pruning_strategies:
            return False
        
        if not 0 < config.sparsity_ratio < 1:
            return False
        
        if config.structured_n <= 0 or config.structured_m <= 0:
            return False
        
        if config.structured_n >= config.structured_m:
            return False
        
        if config.recovery_epochs < 0:
            return False
        
        if config.recovery_learning_rate <= 0:
            return False
        
        return True
    
    def get_supported_types(self) -> List[str]:
        """获取支持的剪枝类型"""
        return [ptype.value for ptype in self.pruning_strategies.keys()]
    
    def get_supported_importance_metrics(self) -> List[str]:
        """获取支持的重要性度量"""
        return list(self.importance_metrics.keys())
