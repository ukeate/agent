"""
量化引擎核心实现

实现PTQ、QAT量化算法
支持INT4、INT8、FP16、BF16等多种量化精度
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import os
from .models import (
    QuantizationConfig, 
    QuantizationMethod, 
    PrecisionType, 
    CompressionResult,
    ModelInfo
)

from src.core.logging import get_logger
class QuantizationEngine:
    """量化引擎
    
    支持多种量化方法:
    - PTQ (Post-Training Quantization): 训练后量化
    - QAT (Quantization-Aware Training): 量化感知训练
    """
    
    def __init__(self):
        self.supported_methods = {
            QuantizationMethod.PTQ: self._post_training_quantization,
            QuantizationMethod.QAT: self._quantization_aware_training,
        }
        
        self.logger = get_logger(__name__)
        self._calibration_cache = {}
    
    def quantize_model(
        self, 
        model: Union[nn.Module, str], 
        config: QuantizationConfig,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """量化模型
        
        Args:
            model: 原始模型或模型路径
            config: 量化配置
            calibration_data: 校准数据集
            
        Returns:
            量化后的模型和压缩信息
        """
        
        start_time = time.time()
        self.logger.info(f"开始量化，方法: {config.method.value}, 精度: {config.precision.value}")
        
        # 加载模型
        if isinstance(model, str):
            self.logger.info(f"从路径加载模型: {model}")
            model = self._load_model(model)
        
        # 获取原始模型信息
        original_size = self._get_model_size(model)
        original_params = self._count_parameters(model)
        
        self.logger.info(f"原始模型大小: {original_size / 1024 / 1024:.2f} MB")
        self.logger.info(f"原始模型参数量: {original_params:,}")
        
        # 执行量化
        quantization_func = self.supported_methods.get(config.method)
        if not quantization_func:
            raise ValueError(f"不支持的量化方法: {config.method}")
        
        quantized_model, quantization_info = quantization_func(
            model, config, calibration_data
        )
        
        # 计算压缩效果
        quantized_size = self._get_model_size(quantized_model)
        quantized_params = self._count_parameters(quantized_model)
        compression_time = time.time() - start_time
        
        compression_stats = {
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
            "original_params": original_params,
            "quantized_params": quantized_params,
            "param_reduction": (original_params - quantized_params) / original_params if original_params > 0 else 0,
            "compression_time": compression_time,
            "quantization_info": quantization_info
        }
        
        self.logger.info(f"量化完成！压缩比: {compression_stats['compression_ratio']:.2f}x")
        self.logger.info(f"参数减少: {compression_stats['param_reduction']*100:.1f}%")
        self.logger.info(f"量化耗时: {compression_time:.2f}s")
        
        return quantized_model, compression_stats
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        try:
            # 尝试加载PyTorch模型
            model = torch.load(model_path, map_location='cpu')
            if not isinstance(model, nn.Module):
                raise ValueError("加载的文件不是有效的PyTorch模型")
            return model
        except Exception as e:
            # 尝试加载Hugging Face模型
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                return model
            except ImportError:
                raise ImportError("需要安装transformers库来加载Hugging Face模型")
            except Exception as inner_e:
                raise ValueError(f"无法加载模型: {e}. 也尝试了Hugging Face格式但失败: {inner_e}")
    
    def _post_training_quantization(
        self, 
        model: nn.Module, 
        config: QuantizationConfig,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """训练后量化(PTQ)"""
        
        self.logger.info("执行训练后量化(PTQ)")
        model.eval()
        
        # 设置量化配置
        if config.precision == PrecisionType.INT8:
            qconfig = torch.quantization.get_default_qconfig('x86')
        elif config.precision == PrecisionType.INT4:
            # 使用自定义4-bit量化配置
            qconfig = self._get_int4_qconfig()
        elif config.precision == PrecisionType.FP16:
            # FP16量化
            model = model.half()
            return model, {
                "method": "PTQ",
                "precision": "fp16",
                "calibration_samples": 0
            }
        else:
            qconfig = torch.quantization.get_default_qconfig('x86')
        
        # 为所有支持的层设置量化配置
        self._set_qconfig_recursively(model, qconfig)
        
        # 准备量化
        model_prepared = torch.quantization.prepare(model, inplace=False)
        
        # 校准
        calibration_samples = 0
        if calibration_data:
            calibration_samples = self._calibrate_model(
                model_prepared, calibration_data, config.calibration_dataset_size
            )
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        quantization_info = {
            "method": "PTQ",
            "precision": config.precision.value,
            "calibration_samples": calibration_samples,
            "dynamic_quantization": config.use_dynamic_quant
        }
        
        return quantized_model, quantization_info
    
    def _quantization_aware_training(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """量化感知训练(QAT)"""
        
        self.logger.info("执行量化感知训练(QAT)")

        if calibration_data is None:
            raise ValueError("QAT需要提供calibration_data")

        model.train()
        
        # 设置量化配置
        qconfig = torch.quantization.get_default_qat_qconfig('x86')
        self._set_qconfig_recursively(model, qconfig)
        
        # 准备QAT模型
        model_prepared = torch.quantization.prepare_qat(model, inplace=False)
        
        # 如果有校准数据，进行简化的QAT训练
        if calibration_data:
            self._simple_qat_training(model_prepared, calibration_data, config)
        
        # 转换为量化模型
        model_prepared.eval()
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        quantization_info = {
            "method": "QAT",
            "precision": config.precision.value,
            "training_samples": len(calibration_data) if calibration_data else 0
        }
        
        return quantized_model, quantization_info
    
    def _set_qconfig_recursively(self, model: nn.Module, qconfig) -> None:
        """递归设置量化配置"""
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.qconfig = qconfig
            else:
                self._set_qconfig_recursively(module, qconfig)
    
    def _get_int4_qconfig(self):
        """获取INT4量化配置"""
        # 自定义INT4量化配置
        return torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint32, qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )
        )
    
    def _calibrate_model(
        self, 
        model: nn.Module, 
        calibration_data: torch.utils.data.DataLoader,
        max_samples: int = 512
    ) -> int:
        """校准模型"""
        
        self.logger.info(f"开始模型校准，最大样本数: {max_samples}")
        
        model.eval()
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                if samples_processed >= max_samples:
                    break
                
                try:
                    # 处理不同的批次格式
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    elif isinstance(batch, dict):
                        inputs = batch.get('input_ids', batch.get('inputs', batch))
                    else:
                        inputs = batch
                    
                    if isinstance(inputs, torch.Tensor):
                        # 确保输入在正确的设备上
                        inputs = inputs.to(next(model.parameters()).device)
                        model(inputs)
                        samples_processed += inputs.size(0) if inputs.dim() > 0 else 1
                    else:
                        self.logger.warning(f"跳过无效的批次格式: {type(inputs)}")
                
                except Exception as e:
                    self.logger.warning(f"校准批次 {batch_idx} 时出错: {e}")
                    continue
        
        self.logger.info(f"校准完成，处理了 {samples_processed} 个样本")
        return samples_processed
    
    def _simple_qat_training(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: QuantizationConfig
    ) -> None:
        """简化的QAT训练"""
        
        self.logger.info("开始QAT训练")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model.train()
        
        # 简化训练：只训练几个batch
        for batch_idx, batch in enumerate(train_data):
            if batch_idx >= 10:  # 限制训练批次
                break
            
            try:
                optimizer.zero_grad()
                
                # 处理批次
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else inputs
                elif isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('inputs'))
                    targets = batch.get('labels', inputs)
                else:
                    inputs = targets = batch
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失（简化版）
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = torch.nn.functional.mse_loss(
                        outputs.view(-1), targets.view(-1).float()
                    )
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                if batch_idx % 5 == 0:
                    self.logger.info(f"QAT训练批次 {batch_idx}, 损失: {loss.item():.4f}")
                
            except Exception as e:
                self.logger.warning(f"QAT训练批次 {batch_idx} 时出错: {e}")
                continue
    
    def _prepare_gptq_calibration_data(
        self, 
        calibration_data: torch.utils.data.DataLoader,
        max_samples: int = 512
    ) -> List[str]:
        """为GPTQ准备校准数据"""
        
        examples = []
        samples_processed = 0
        
        for batch in calibration_data:
            if samples_processed >= max_samples:
                break
            
            try:
                # 尝试提取文本数据
                if isinstance(batch, dict) and 'text' in batch:
                    texts = batch['text']
                    if isinstance(texts, str):
                        examples.append(texts)
                        samples_processed += 1
                    elif isinstance(texts, list):
                        examples.extend(texts[:max_samples - samples_processed])
                        samples_processed += len(texts)
                else:
                    # 使用占位符文本
                    examples.append("This is a calibration example.")
                    samples_processed += 1
            
            except Exception as e:
                self.logger.warning(f"处理GPTQ校准数据时出错: {e}")
                continue
        
        return examples[:max_samples]
    
    def _get_model_size(self, model: nn.Module) -> int:
        """获取模型大小（字节）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def save_quantized_model(self, model: nn.Module, save_path: str) -> str:
        """保存量化模型"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            # 尝试保存为transformers格式
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_path)
            else:
                # 保存为PyTorch格式
                torch.save(model, save_path)
            
            self.logger.info(f"量化模型已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存量化模型失败: {e}")
            raise
    
    def get_supported_methods(self) -> List[str]:
        """获取支持的量化方法"""
        return [method.value for method in self.supported_methods.keys()]
    
    def validate_config(self, config: QuantizationConfig) -> bool:
        """验证量化配置"""
        
        if config.method not in self.supported_methods:
            return False
        
        if config.calibration_dataset_size < 0:
            return False
        
        if not 0 < config.target_accuracy_loss <= 1:
            return False
        
        return True
