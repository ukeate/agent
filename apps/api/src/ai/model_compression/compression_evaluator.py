"""
压缩效果评估系统

提供模型大小、推理延迟、内存占用的综合评估
实现压缩前后的性能对比分析
支持不同硬件平台的性能测试
提供压缩策略推荐和优化建议
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
import time
import psutil
import os
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import gc
import threading
from contextlib import contextmanager

from .models import (
    CompressionResult, 
    HardwareBenchmark, 
    CompressionStrategy,
    ModelInfo,
    DEFAULT_COMPRESSION_STRATEGIES
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    model_size_mb: float
    parameter_count: int
    inference_latency_ms: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    accuracy_score: Optional[float] = None
    perplexity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    """对比报告"""
    original_metrics: EvaluationMetrics
    compressed_metrics: EvaluationMetrics
    compression_ratio: float
    speedup_ratio: float
    memory_reduction: float
    accuracy_retention: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CompressionEvaluator:
    """压缩评估系统
    
    提供全面的模型压缩效果评估:
    - 模型大小和参数量分析
    - 推理性能基准测试
    - 内存使用情况监控
    - 精度保持度评估
    - 硬件兼容性测试
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmark_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 性能监控配置
        self.warmup_runs = 10
        self.benchmark_runs = 50
        self.memory_monitor_interval = 0.1
        
        # 支持的评估任务
        self.evaluation_tasks = {
            "text_generation": self._evaluate_text_generation,
            "classification": self._evaluate_classification,
            "question_answering": self._evaluate_question_answering
        }
    
    def evaluate_model(
        self,
        model: nn.Module,
        model_name: str = "model",
        test_data: Optional[torch.utils.data.DataLoader] = None,
        task_type: str = "text_generation",
        batch_sizes: List[int] = None
    ) -> EvaluationMetrics:
        """评估单个模型性能"""
        
        self.logger.info(f"开始评估模型: {model_name}")
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        
        # 移动模型到设备
        model = model.to(self.device)
        model.eval()
        
        # 基础指标
        model_size = self._calculate_model_size(model)
        param_count = self._count_parameters(model)
        
        # 性能基准测试
        latency_results = []
        throughput_results = []
        memory_results = []
        
        for batch_size in batch_sizes:
            self.logger.info(f"测试批次大小: {batch_size}")
            
            # 创建测试输入
            test_input = self._create_test_input(model, batch_size)
            
            # 延迟测试
            latency = self._benchmark_latency(model, test_input)
            latency_results.append(latency)
            
            # 吞吐量测试
            throughput = self._benchmark_throughput(model, test_input, batch_size)
            throughput_results.append(throughput)
            
            # 内存使用测试
            memory_usage = self._benchmark_memory_usage(model, test_input)
            memory_results.append(memory_usage)
        
        # 取平均值或选择最优结果
        avg_latency = np.mean(latency_results)
        max_throughput = max(throughput_results)
        peak_memory = max(memory_results)
        
        # 精度评估（如果有测试数据）
        accuracy_score = None
        perplexity = None
        if test_data is not None:
            evaluation_func = self.evaluation_tasks.get(task_type)
            if evaluation_func:
                accuracy_score, perplexity = evaluation_func(model, test_data)
        
        metrics = EvaluationMetrics(
            model_size_mb=model_size,
            parameter_count=param_count,
            inference_latency_ms=avg_latency,
            memory_usage_mb=peak_memory,
            throughput_tokens_per_sec=max_throughput,
            accuracy_score=accuracy_score,
            perplexity=perplexity
        )
        
        self.logger.info(f"模型评估完成: {model_name}")
        self._log_metrics(metrics)
        
        return metrics
    
    def compare_models(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: Optional[torch.utils.data.DataLoader] = None,
        task_type: str = "text_generation"
    ) -> ComparisonReport:
        """对比原始模型和压缩模型"""
        
        self.logger.info("开始模型对比评估")
        
        # 评估原始模型
        original_metrics = self.evaluate_model(
            original_model, "original", test_data, task_type
        )
        
        # 评估压缩模型
        compressed_metrics = self.evaluate_model(
            compressed_model, "compressed", test_data, task_type
        )
        
        # 计算对比指标
        compression_ratio = (
            original_metrics.model_size_mb / compressed_metrics.model_size_mb
            if compressed_metrics.model_size_mb > 0 else 0
        )
        
        speedup_ratio = (
            original_metrics.inference_latency_ms / compressed_metrics.inference_latency_ms
            if compressed_metrics.inference_latency_ms > 0 else 0
        )
        
        memory_reduction = (
            (original_metrics.memory_usage_mb - compressed_metrics.memory_usage_mb) 
            / original_metrics.memory_usage_mb
            if original_metrics.memory_usage_mb > 0 else 0
        )
        
        accuracy_retention = None
        if (original_metrics.accuracy_score is not None and 
            compressed_metrics.accuracy_score is not None):
            accuracy_retention = (
                compressed_metrics.accuracy_score / original_metrics.accuracy_score
                if original_metrics.accuracy_score > 0 else 0
            )
        
        report = ComparisonReport(
            original_metrics=original_metrics,
            compressed_metrics=compressed_metrics,
            compression_ratio=compression_ratio,
            speedup_ratio=speedup_ratio,
            memory_reduction=memory_reduction,
            accuracy_retention=accuracy_retention
        )
        
        self.logger.info(f"模型对比完成 - 压缩比: {compression_ratio:.2f}x, 加速比: {speedup_ratio:.2f}x")
        
        return report
    
    def benchmark_hardware_performance(
        self,
        model: nn.Module,
        device_name: str = None,
        sequence_lengths: List[int] = None,
        batch_sizes: List[int] = None
    ) -> List[HardwareBenchmark]:
        """硬件性能基准测试"""
        
        if device_name is None:
            device_name = str(self.device)
        
        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024]
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        self.logger.info(f"开始硬件性能基准测试: {device_name}")
        
        model = model.to(self.device)
        model.eval()
        
        benchmarks = []
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                self.logger.info(f"测试配置: batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    # 创建测试输入
                    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                        vocab_size = model.config.vocab_size
                    else:
                        vocab_size = 50000  # 默认词汇表大小
                    
                    test_input = torch.randint(
                        0, vocab_size, (batch_size, seq_len), 
                        device=self.device
                    )
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(self.warmup_runs):
                            _ = model(test_input)
                    
                    # 性能测试
                    latencies = []
                    memory_usages = []
                    
                    with torch.no_grad():
                        for _ in range(self.benchmark_runs):
                            # 内存监控
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                memory_before = torch.cuda.memory_allocated()
                            
                            # 延迟测试
                            start_time = time.time()
                            _ = model(test_input)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            end_time = time.time()
                            
                            latency_ms = (end_time - start_time) * 1000
                            latencies.append(latency_ms)
                            
                            # 内存使用
                            if torch.cuda.is_available():
                                memory_after = torch.cuda.memory_allocated()
                                memory_usage = (memory_after - memory_before) / 1024 / 1024
                                memory_usages.append(memory_usage)
                    
                    # 计算统计信息
                    latency_p50 = np.percentile(latencies, 50)
                    latency_p95 = np.percentile(latencies, 95)
                    latency_p99 = np.percentile(latencies, 99)
                    avg_memory = np.mean(memory_usages) if memory_usages else 0
                    
                    throughput = (batch_size * seq_len) / (latency_p50 / 1000)  # tokens/second
                    
                    device_type = "gpu" if torch.cuda.is_available() else "cpu"
                    
                    benchmark = HardwareBenchmark(
                        device_name=device_name,
                        device_type=device_type,
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        throughput=throughput,
                        latency_p50=latency_p50,
                        latency_p95=latency_p95,
                        latency_p99=latency_p99,
                        memory_usage=avg_memory,
                        power_consumption=None  # 需要专门的硬件监控
                    )
                    
                    benchmarks.append(benchmark)
                
                except Exception as e:
                    self.logger.warning(f"基准测试失败 batch_size={batch_size}, seq_len={seq_len}: {e}")
                    continue
        
        self.logger.info(f"硬件性能基准测试完成，共{len(benchmarks)}个配置")
        
        return benchmarks
    
    def recommend_compression_strategy(
        self,
        model_info: ModelInfo,
        target_scenario: str = "cloud",
        accuracy_tolerance: float = 0.05,
        size_reduction_target: float = 0.5
    ) -> List[CompressionStrategy]:
        """推荐压缩策略"""
        
        self.logger.info(f"为{target_scenario}场景推荐压缩策略")
        
        suitable_strategies = []
        
        for strategy in DEFAULT_COMPRESSION_STRATEGIES:
            # 过滤目标场景
            if strategy.target_scenario != target_scenario:
                continue
            
            # 检查精度要求
            expected_accuracy_loss = 1 - strategy.expected_accuracy_retention
            if expected_accuracy_loss > accuracy_tolerance:
                continue
            
            # 检查压缩目标
            expected_size_reduction = 1 - (1 / strategy.expected_compression_ratio)
            if expected_size_reduction < size_reduction_target:
                continue
            
            suitable_strategies.append(strategy)
        
        # 按推荐度排序（综合考虑压缩比和精度保持）
        suitable_strategies.sort(
            key=lambda s: s.expected_compression_ratio * s.expected_accuracy_retention,
            reverse=True
        )
        
        self.logger.info(f"找到{len(suitable_strategies)}个推荐策略")
        
        return suitable_strategies
    
    def generate_evaluation_report(
        self,
        comparison_report: ComparisonReport,
        save_path: str = None
    ) -> Dict[str, Any]:
        """生成详细的评估报告"""
        
        report = {
            "evaluation_summary": {
                "compression_ratio": f"{comparison_report.compression_ratio:.2f}x",
                "speedup_ratio": f"{comparison_report.speedup_ratio:.2f}x",
                "memory_reduction": f"{comparison_report.memory_reduction*100:.1f}%",
                "accuracy_retention": f"{comparison_report.accuracy_retention*100:.1f}%" 
                    if comparison_report.accuracy_retention else "N/A"
            },
            "original_model": comparison_report.original_metrics.to_dict(),
            "compressed_model": comparison_report.compressed_metrics.to_dict(),
            "recommendations": self._generate_recommendations(comparison_report),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size_bytes = param_size + buffer_size
        return total_size_bytes / (1024 * 1024)  # Convert to MB
    
    def _count_parameters(self, model: nn.Module) -> int:
        """计算参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _create_test_input(self, model: nn.Module, batch_size: int = 1) -> torch.Tensor:
        """创建测试输入"""
        
        # 尝试从模型配置获取信息
        if hasattr(model, 'config'):
            config = model.config
            if hasattr(config, 'vocab_size') and hasattr(config, 'max_position_embeddings'):
                vocab_size = config.vocab_size
                max_seq_len = min(config.max_position_embeddings, 512)
            else:
                vocab_size = 50000
                max_seq_len = 512
        else:
            vocab_size = 50000
            max_seq_len = 512
        
        # 创建随机输入
        seq_len = min(max_seq_len, 256)  # 使用较短的序列进行测试
        test_input = torch.randint(
            0, vocab_size, (batch_size, seq_len), 
            device=self.device
        )
        
        return test_input
    
    def _benchmark_latency(self, model: nn.Module, test_input: torch.Tensor) -> float:
        """基准测试延迟"""
        
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(test_input)
        
        # 测试延迟
        latencies = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(test_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        return np.mean(latencies)
    
    def _benchmark_throughput(
        self, 
        model: nn.Module, 
        test_input: torch.Tensor, 
        batch_size: int
    ) -> float:
        """基准测试吞吐量"""
        
        seq_len = test_input.size(1)
        latency_ms = self._benchmark_latency(model, test_input)
        
        # 计算每秒处理的token数
        tokens_per_batch = batch_size * seq_len
        throughput = tokens_per_batch / (latency_ms / 1000)
        
        return throughput
    
    @contextmanager
    def _memory_monitor(self):
        """内存监控上下文管理器"""
        
        memory_usage = []
        stop_monitoring = threading.Event()
        
        def monitor():
            while not stop_monitoring.is_set():
                if torch.cuda.is_available():
                    memory = torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage.append(memory)
                time.sleep(self.memory_monitor_interval)
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            yield memory_usage
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)
    
    def _benchmark_memory_usage(self, model: nn.Module, test_input: torch.Tensor) -> float:
        """基准测试内存使用"""
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        with self._memory_monitor() as memory_usage:
            with torch.no_grad():
                for _ in range(5):  # 少量测试以监控内存峰值
                    _ = model(test_input)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
        
        return max(memory_usage) if memory_usage else 0.0
    
    def _evaluate_text_generation(
        self, 
        model: nn.Module, 
        test_data: torch.utils.data.DataLoader
    ) -> Tuple[Optional[float], Optional[float]]:
        """评估文本生成任务"""
        
        model.eval()
        total_loss = 0.0
        num_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                if batch_idx >= 100:  # 限制评估批次
                    break
                
                try:
                    # 处理不同的批次格式
                    if isinstance(batch, dict):
                        inputs = batch.get('input_ids', batch.get('inputs'))
                        labels = batch.get('labels', inputs)
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        labels = batch[1] if len(batch) > 1 else inputs
                    else:
                        inputs = labels = batch
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs, labels=labels)
                    
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        # 计算token数量（用于perplexity）
                        if hasattr(labels, 'numel'):
                            num_tokens += labels.numel()
                    
                    num_batches += 1
                
                except Exception as e:
                    self.logger.warning(f"评估批次 {batch_idx} 时出错: {e}")
                    continue
        
        if num_batches == 0:
            return None, None
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss) if avg_loss > 0 else None
        
        # 对于生成任务，使用perplexity的倒数作为accuracy的近似
        accuracy = 1.0 / perplexity if perplexity and perplexity > 1 else None
        
        return accuracy, perplexity
    
    def _evaluate_classification(
        self, 
        model: nn.Module, 
        test_data: torch.utils.data.DataLoader
    ) -> Tuple[Optional[float], Optional[float]]:
        """评估分类任务"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                if batch_idx >= 100:  # 限制评估批次
                    break
                
                try:
                    if isinstance(batch, dict):
                        inputs = batch.get('input_ids', batch.get('inputs'))
                        labels = batch.get('labels')
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        labels = batch[1] if len(batch) > 1 else None
                    else:
                        inputs = batch
                        labels = None
                    
                    if labels is None:
                        continue
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs)
                    
                    if hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                
                except Exception as e:
                    self.logger.warning(f"评估批次 {batch_idx} 时出错: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else None
        return accuracy, None
    
    def _evaluate_question_answering(
        self, 
        model: nn.Module, 
        test_data: torch.utils.data.DataLoader
    ) -> Tuple[Optional[float], Optional[float]]:
        """评估问答任务"""
        
        # 简化实现：使用分类评估
        return self._evaluate_classification(model, test_data)
    
    def _generate_recommendations(self, comparison_report: ComparisonReport) -> List[str]:
        """生成优化建议"""
        
        recommendations = []
        
        # 压缩效果分析
        if comparison_report.compression_ratio < 2.0:
            recommendations.append("压缩比较低，建议尝试更激进的量化策略（如INT4）或增加剪枝比例")
        
        if comparison_report.speedup_ratio < 1.2:
            recommendations.append("加速效果不明显，建议优化推理引擎或使用结构化剪枝")
        
        # 精度保持分析
        if comparison_report.accuracy_retention and comparison_report.accuracy_retention < 0.9:
            recommendations.append("精度损失较大，建议减少压缩强度或使用知识蒸馏恢复精度")
        elif comparison_report.accuracy_retention and comparison_report.accuracy_retention > 0.98:
            recommendations.append("精度保持良好，可以尝试更高的压缩比以获得更大的模型压缩")
        
        # 内存使用分析
        if comparison_report.memory_reduction < 0.3:
            recommendations.append("内存减少有限，建议使用更低精度的量化或增加剪枝力度")
        
        # 综合建议
        if (comparison_report.compression_ratio > 4.0 and 
            comparison_report.accuracy_retention and 
            comparison_report.accuracy_retention > 0.95):
            recommendations.append("压缩效果优秀，可以考虑部署到生产环境")
        
        return recommendations
    
    def _log_metrics(self, metrics: EvaluationMetrics) -> None:
        """记录评估指标"""
        
        self.logger.info("=== 模型评估指标 ===")
        self.logger.info(f"模型大小: {metrics.model_size_mb:.2f} MB")
        self.logger.info(f"参数数量: {metrics.parameter_count:,}")
        self.logger.info(f"推理延迟: {metrics.inference_latency_ms:.2f} ms")
        self.logger.info(f"内存使用: {metrics.memory_usage_mb:.2f} MB")
        self.logger.info(f"吞吐量: {metrics.throughput_tokens_per_sec:.2f} tokens/s")
        
        if metrics.accuracy_score is not None:
            self.logger.info(f"准确率: {metrics.accuracy_score:.4f}")
        
        if metrics.perplexity is not None:
            self.logger.info(f"困惑度: {metrics.perplexity:.2f}")
    
    def _save_report(self, report: Dict[str, Any], save_path: str) -> None:
        """保存评估报告"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估报告已保存到: {save_path}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        
        device_info = {
            "device_type": str(self.device),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if torch.cuda.is_available():
            device_info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
            })
        
        return device_info