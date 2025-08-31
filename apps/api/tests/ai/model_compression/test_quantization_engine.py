"""
量化引擎测试

测试量化引擎的各种功能和边界条件
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.ai.model_compression.quantization_engine import QuantizationEngine
from src.ai.model_compression.models import (
    QuantizationConfig,
    QuantizationMethod,
    PrecisionType
)


class SimpleTestModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@pytest.fixture
def simple_model():
    """创建简单测试模型"""
    return SimpleTestModel()


@pytest.fixture
def quantization_engine():
    """创建量化引擎"""
    return QuantizationEngine()


@pytest.fixture
def basic_config():
    """基础量化配置"""
    return QuantizationConfig(
        method=QuantizationMethod.PTQ,
        precision=PrecisionType.INT8,
        calibration_dataset_size=100
    )


class TestQuantizationEngine:
    """量化引擎测试类"""
    
    def test_engine_initialization(self, quantization_engine):
        """测试引擎初始化"""
        assert quantization_engine is not None
        assert len(quantization_engine.supported_methods) > 0
        assert QuantizationMethod.PTQ in quantization_engine.supported_methods
    
    def test_model_size_calculation(self, quantization_engine, simple_model):
        """测试模型大小计算"""
        size = quantization_engine._get_model_size(simple_model)
        assert size > 0
        assert isinstance(size, int)
    
    def test_parameter_counting(self, quantization_engine, simple_model):
        """测试参数计数"""
        count = quantization_engine._count_parameters(simple_model)
        expected_count = 10 * 5 + 5 + 5 * 2 + 2  # weights + biases
        assert count == expected_count
    
    def test_config_validation(self, quantization_engine):
        """测试配置验证"""
        # 有效配置
        valid_config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
        assert quantization_engine.validate_config(valid_config) is True
        
        # 无效配置
        invalid_config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8,
            calibration_dataset_size=-1
        )
        assert quantization_engine.validate_config(invalid_config) is False
    
    def test_ptq_quantization(self, quantization_engine, simple_model, basic_config):
        """测试PTQ量化"""
        original_size = quantization_engine._get_model_size(simple_model)
        
        # 执行量化
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, basic_config
        )
        
        # 验证结果
        assert quantized_model is not None
        assert stats is not None
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] >= 1.0
        
        # 验证量化后的模型可以运行
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = quantized_model(test_input)
            assert output is not None
            assert output.shape == (1, 2)
    
    def test_fp16_quantization(self, quantization_engine, simple_model):
        """测试FP16量化"""
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.FP16
        )
        
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, config
        )
        
        # 验证FP16模型
        assert quantized_model is not None
        
        # 检查参数数据类型
        for param in quantized_model.parameters():
            assert param.dtype == torch.float16
    
    def test_gptq_quantization_fallback(self, quantization_engine, simple_model):
        """测试GPTQ量化（降级到PTQ）"""
        config = QuantizationConfig(
            method=QuantizationMethod.GPTQ,
            precision=PrecisionType.INT4
        )
        
        # 由于没有安装gptqmodel，应该降级到伪实现
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, config
        )
        
        assert quantized_model is not None
        assert stats is not None
        assert "method" in stats["quantization_info"]
    
    def test_int4_quantization(self, quantization_engine, simple_model):
        """测试INT4量化"""
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT4
        )
        
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, config
        )
        
        assert quantized_model is not None
        assert stats["compression_ratio"] > 1.0
    
    def test_with_calibration_data(self, quantization_engine, simple_model, basic_config):
        """测试使用校准数据的量化"""
        # 创建模拟校准数据
        calibration_data = []
        for _ in range(5):
            batch = torch.randn(2, 10)
            calibration_data.append((batch,))
        
        # 创建DataLoader模拟
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(calibration_data))
        mock_dataloader.__len__ = Mock(return_value=len(calibration_data))
        
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, basic_config, mock_dataloader
        )
        
        assert quantized_model is not None
        assert stats is not None
        assert stats["quantization_info"]["calibration_samples"] > 0
    
    def test_quantize_tensor_int8(self, quantization_engine):
        """测试INT8张量量化"""
        tensor = torch.randn(3, 3) * 10  # 创建一些范围较大的值
        quantized = quantization_engine._quantize_tensor_int8(tensor)
        
        assert quantized.shape == tensor.shape
        assert quantized.dtype == tensor.dtype
        
        # 验证量化后的值在合理范围内
        assert torch.all(torch.abs(quantized) <= torch.abs(tensor) + 1e-3)
    
    def test_quantize_tensor_int4(self, quantization_engine):
        """测试INT4张量量化"""
        tensor = torch.randn(3, 3) * 5
        quantized = quantization_engine._quantize_tensor_int4(tensor)
        
        assert quantized.shape == tensor.shape
        assert quantized.dtype == tensor.dtype
    
    def test_save_quantized_model(self, quantization_engine, simple_model, tmp_path):
        """测试保存量化模型"""
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
        
        quantized_model, _ = quantization_engine.quantize_model(simple_model, config)
        
        save_path = tmp_path / "quantized_model.pth"
        saved_path = quantization_engine.save_quantized_model(quantized_model, str(save_path))
        
        assert saved_path == str(save_path)
        assert save_path.exists()
        
        # 验证能够加载保存的模型
        loaded_model = torch.load(save_path)
        assert loaded_model is not None
    
    def test_get_supported_methods(self, quantization_engine):
        """测试获取支持的方法"""
        methods = quantization_engine.get_supported_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert "post_training_quantization" in methods
    
    def test_error_handling_invalid_model_path(self, quantization_engine, basic_config):
        """测试无效模型路径的错误处理"""
        with pytest.raises(ValueError, match="无法加载模型"):
            quantization_engine.quantize_model("invalid_path.pth", basic_config)
    
    def test_error_handling_unsupported_method(self, quantization_engine, simple_model):
        """测试不支持的量化方法"""
        # 创建一个无效的量化方法配置
        invalid_config = Mock()
        invalid_config.method = "invalid_method"
        
        # 临时替换supported_methods来测试错误处理
        original_methods = quantization_engine.supported_methods
        quantization_engine.supported_methods = {}
        
        try:
            with pytest.raises(ValueError, match="不支持的量化方法"):
                quantization_engine.quantize_model(simple_model, invalid_config)
        finally:
            quantization_engine.supported_methods = original_methods
    
    def test_calibration_with_different_batch_formats(self, quantization_engine, simple_model):
        """测试不同批次格式的校准"""
        # 测试字典格式
        dict_batch = {"input_ids": torch.randn(1, 10), "labels": torch.randn(1, 2)}
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([dict_batch]))
        
        # 执行校准
        samples = quantization_engine._calibrate_model(simple_model, mock_dataloader, 10)
        assert samples > 0
        
        # 测试张量格式
        tensor_batch = torch.randn(2, 10)
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([tensor_batch]))
        
        samples = quantization_engine._calibrate_model(simple_model, mock_dataloader, 10)
        assert samples > 0
    
    def test_model_loading_from_path(self, quantization_engine, simple_model, tmp_path):
        """测试从路径加载模型"""
        # 保存模型到临时文件
        model_path = tmp_path / "test_model.pth"
        torch.save(simple_model, model_path)
        
        # 使用路径加载模型进行量化
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
        
        quantized_model, stats = quantization_engine.quantize_model(str(model_path), config)
        
        assert quantized_model is not None
        assert stats is not None
    
    @patch('src.ai.model_compression.quantization_engine.torch.quantization.prepare')
    def test_quantization_error_handling(self, mock_prepare, quantization_engine, simple_model, basic_config):
        """测试量化过程中的错误处理"""
        # 模拟量化准备失败
        mock_prepare.side_effect = RuntimeError("Quantization failed")
        
        with pytest.raises(RuntimeError):
            quantization_engine._post_training_quantization(simple_model, basic_config)
    
    def test_weight_quantization_methods(self, quantization_engine, simple_model):
        """测试权重量化方法"""
        # 测试FP16量化
        fp16_model = quantization_engine._apply_weight_quantization(simple_model, PrecisionType.FP16)
        for param in fp16_model.parameters():
            assert param.dtype == torch.float16
        
        # 测试INT8量化
        int8_model = quantization_engine._apply_weight_quantization(simple_model, PrecisionType.INT8)
        assert int8_model is not None
        
        # 测试INT4量化
        int4_model = quantization_engine._apply_weight_quantization(simple_model, PrecisionType.INT4)
        assert int4_model is not None
    
    def test_concurrent_quantization(self, quantization_engine, simple_model, basic_config):
        """测试并发量化（简单测试）"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def quantize_worker():
            try:
                model_copy = SimpleTestModel()
                result = quantization_engine.quantize_model(model_copy, basic_config)
                results.put(("success", result))
            except Exception as e:
                results.put(("error", str(e)))
        
        # 启动多个线程
        threads = []
        for _ in range(2):
            thread = threading.Thread(target=quantize_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = 0
        while not results.empty():
            status, _ = results.get()
            if status == "success":
                success_count += 1
        
        assert success_count == 2


@pytest.mark.parametrize("method,precision", [
    (QuantizationMethod.PTQ, PrecisionType.INT8),
    (QuantizationMethod.PTQ, PrecisionType.FP16),
    (QuantizationMethod.QAT, PrecisionType.INT8),
])
def test_different_quantization_combinations(method, precision, quantization_engine, simple_model):
    """参数化测试不同的量化组合"""
    config = QuantizationConfig(method=method, precision=precision)
    
    quantized_model, stats = quantization_engine.quantize_model(simple_model, config)
    
    assert quantized_model is not None
    assert stats is not None
    assert stats["compression_ratio"] >= 1.0


class TestQuantizationEdgeCases:
    """量化边界情况测试"""
    
    def test_empty_model(self, quantization_engine, basic_config):
        """测试空模型"""
        empty_model = nn.Module()
        
        quantized_model, stats = quantization_engine.quantize_model(empty_model, basic_config)
        
        assert quantized_model is not None
        assert stats["original_params"] == 0
    
    def test_single_layer_model(self, quantization_engine, basic_config):
        """测试单层模型"""
        single_layer = nn.Linear(5, 1)
        
        quantized_model, stats = quantization_engine.quantize_model(single_layer, basic_config)
        
        assert quantized_model is not None
        assert stats["original_params"] == 6  # 5*1 + 1 bias
    
    def test_very_large_calibration_size(self, quantization_engine, simple_model):
        """测试非常大的校准数据集大小"""
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8,
            calibration_dataset_size=10000  # 很大的数字
        )
        
        # 只提供少量数据
        small_data = [torch.randn(1, 10) for _ in range(3)]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(small_data))
        
        quantized_model, stats = quantization_engine.quantize_model(
            simple_model, config, mock_dataloader
        )
        
        # 应该只处理实际可用的数据量
        assert quantized_model is not None
        assert stats["quantization_info"]["calibration_samples"] <= len(small_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])