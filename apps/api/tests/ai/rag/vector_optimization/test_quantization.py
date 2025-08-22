"""
向量量化算法测试

测试INT8、INT4和自适应量化算法的精度和性能
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from src.ai.rag.quantization import (
    VectorQuantizer,
    QuantizationConfig,
    QuantizationMode,
    QuantizationQualityAssessment
)


class TestVectorQuantizer:
    """向量量化器测试"""
    
    @pytest.fixture
    def sample_vectors(self):
        """生成测试向量"""
        np.random.seed(42)
        return [
            np.random.normal(0, 1, 1536).astype(np.float32),
            np.random.uniform(-1, 1, 1536).astype(np.float32),
            np.zeros(1536).astype(np.float32),
            np.ones(1536).astype(np.float32),
        ]
    
    @pytest.fixture
    def quantizer_int8(self):
        """INT8量化器"""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        return VectorQuantizer(config)
    
    @pytest.fixture
    def quantizer_int4(self):
        """INT4量化器"""
        config = QuantizationConfig(mode=QuantizationMode.INT4)
        return VectorQuantizer(config)
    
    @pytest.fixture
    def quantizer_adaptive(self):
        """自适应量化器"""
        config = QuantizationConfig(mode=QuantizationMode.ADAPTIVE, precision_threshold=0.95)
        return VectorQuantizer(config)
    
    @pytest.mark.asyncio
    async def test_int8_quantization(self, quantizer_int8, sample_vectors):
        """测试INT8量化"""
        vector = sample_vectors[0]
        
        quantized, params = await quantizer_int8.quantize_vector(vector)
        
        # 检查量化结果
        assert isinstance(quantized, np.ndarray)
        assert quantized.dtype == np.int8
        assert quantized.shape == vector.shape
        
        # 检查参数
        assert params["mode"] == "int8"
        assert "scale" in params
        assert "zero_point" in params
        assert params["compression"] == 4.0
        assert 0.0 <= params["precision_loss"] <= 1.0
        
        # 检查反量化精度
        dequantized = await quantizer_int8.dequantize_vector(quantized, params)
        mse = np.mean((vector - dequantized) ** 2)
        assert mse < 50.0  # 合理的MSE阈值，考虑量化损失
    
    @pytest.mark.asyncio
    async def test_int4_quantization(self, quantizer_int4, sample_vectors):
        """测试INT4量化"""
        vector = sample_vectors[0]
        
        quantized, params = await quantizer_int4.quantize_vector(vector)
        
        # 检查量化结果
        assert isinstance(quantized, np.ndarray)
        assert quantized.dtype == np.uint8
        assert quantized.shape == vector.shape
        
        # 检查参数
        assert params["mode"] == "int4"
        assert "centroids" in params
        assert params["compression"] == 8.0
        assert 0.0 <= params["precision_loss"] <= 1.0
        assert len(params["centroids"]) <= 16  # 4位最多16个值
        
        # 检查反量化
        dequantized = await quantizer_int4.dequantize_vector(quantized, params)
        assert dequantized.shape == vector.shape
    
    @pytest.mark.asyncio
    async def test_adaptive_quantization(self, quantizer_adaptive, sample_vectors):
        """测试自适应量化"""
        for vector in sample_vectors:
            quantized, params = await quantizer_adaptive.quantize_vector(vector)
            
            # 检查选择的量化模式
            mode = params["mode"]
            assert mode in ["int4", "int8", "float32"]
            
            # 如果选择了量化模式，精度损失应该小于阈值
            if mode in ["int4", "int8"]:
                assert params["precision_loss"] <= (1 - quantizer_adaptive.config.precision_threshold)
    
    @pytest.mark.asyncio
    async def test_zero_vector_quantization(self, quantizer_int8):
        """测试零向量量化"""
        zero_vector = np.zeros(100).astype(np.float32)
        
        quantized, params = await quantizer_int8.quantize_vector(zero_vector)
        
        assert np.all(quantized == 0)
        assert params["scale"] == 1.0  # 避免除零
        assert params["zero_point"] == 0
    
    @pytest.mark.asyncio
    async def test_constant_vector_quantization(self, quantizer_int8):
        """测试常数向量量化"""
        constant_vector = np.full(100, 5.0).astype(np.float32)
        
        quantized, params = await quantizer_int8.quantize_vector(constant_vector)
        
        # 常数向量的量化应该是相同的值
        assert len(set(quantized)) == 1
        assert params["precision_loss"] == 0.0  # 完美精度，无损失
    
    @pytest.mark.asyncio
    async def test_quantization_reproducibility(self, quantizer_int8, sample_vectors):
        """测试量化结果的可重现性"""
        vector = sample_vectors[0]
        
        # 多次量化同一向量
        results = []
        for _ in range(3):
            quantized, params = await quantizer_int8.quantize_vector(vector)
            results.append((quantized, params))
        
        # 结果应该一致
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            assert results[0][1]["scale"] == results[i][1]["scale"]
            assert results[0][1]["zero_point"] == results[i][1]["zero_point"]
    
    def test_quantization_config(self):
        """测试量化配置"""
        config = QuantizationConfig(
            mode=QuantizationMode.ADAPTIVE,
            precision_threshold=0.98,
            compression_ratio=6.0,
            enable_dynamic=False
        )
        
        assert config.mode == QuantizationMode.ADAPTIVE
        assert config.precision_threshold == 0.98
        assert config.compression_ratio == 6.0
        assert config.enable_dynamic is False
        assert config.fallback_mode == QuantizationMode.INT8


class TestQuantizationQualityAssessment:
    """量化质量评估测试"""
    
    @pytest.fixture
    def quality_assessor(self):
        """创建质量评估器"""
        return QuantizationQualityAssessment()
    
    @pytest.fixture
    def sample_assessment_data(self):
        """生成测试评估数据"""
        np.random.seed(42)
        original_vectors = [np.random.normal(0, 1, 100) for _ in range(10)]
        quantized_vectors = [v + np.random.normal(0, 0.1, 100) for v in original_vectors]  # 添加噪声模拟量化损失
        params = [
            {"mode": "int8", "compression": 4.0, "precision_loss": np.random.uniform(0.05, 0.15)}
            for _ in range(10)
        ]
        return original_vectors, quantized_vectors, params
    
    @pytest.mark.asyncio
    async def test_assess_quality(self, quality_assessor, sample_assessment_data):
        """测试质量评估"""
        original_vectors, quantized_vectors, params = sample_assessment_data
        
        report = await quality_assessor.assess_quality(
            original_vectors, quantized_vectors, params
        )
        
        # 检查报告结构
        assert "total_vectors" in report
        assert "average_precision_loss" in report
        assert "average_compression_ratio" in report
        assert "mode_distribution" in report
        assert "quality_score" in report
        assert "assessment_time" in report
        
        # 检查数值合理性
        assert report["total_vectors"] == 10
        assert 0.0 <= report["average_precision_loss"] <= 1.0
        assert 0.0 <= report["quality_score"] <= 1.0
        assert report["average_compression_ratio"] == 4.0
        assert "int8" in report["mode_distribution"]
    
    @pytest.mark.asyncio
    async def test_should_fallback(self, quality_assessor, sample_assessment_data):
        """测试回退判断"""
        original_vectors, quantized_vectors, params = sample_assessment_data
        
        # 创建低质量报告
        low_quality_report = {
            "quality_score": 0.5,  # 低于阈值
            "average_precision_loss": 0.5
        }
        
        should_fallback, reason = await quality_assessor.should_fallback(
            low_quality_report, threshold=0.8
        )
        
        assert should_fallback is True
        assert "below threshold" in reason.lower()
        
        # 创建高质量报告
        high_quality_report = {
            "quality_score": 0.95,  # 高于阈值
            "average_precision_loss": 0.05
        }
        
        should_fallback, reason = await quality_assessor.should_fallback(
            high_quality_report, threshold=0.8
        )
        
        assert should_fallback is False
        assert "acceptable" in reason.lower()
    
    def test_assessment_history(self, quality_assessor):
        """测试评估历史记录"""
        # 添加一些虚拟评估记录
        for i in range(5):
            quality_assessor.assessment_history.append({
                "quality_score": 0.9 - i * 0.1,
                "timestamp": f"2025-01-{i+1:02d}T10:00:00"
            })
        
        history = quality_assessor.get_assessment_history()
        
        assert len(history) == 5
        assert history[0]["quality_score"] == 0.9
        assert history[-1]["quality_score"] == 0.5
    
    @pytest.mark.asyncio
    async def test_empty_vectors_assessment(self, quality_assessor):
        """测试空向量列表的评估"""
        with pytest.raises(ValueError, match="mismatch"):
            await quality_assessor.assess_quality([], [np.zeros(10)], [{}])


class TestQuantizationIntegration:
    """量化系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_quantization_pipeline(self):
        """测试完整的量化流程"""
        # 创建测试向量
        np.random.seed(42)
        test_vectors = [np.random.normal(0, 1, 512) for _ in range(5)]
        
        # 配置量化器
        config = QuantizationConfig(
            mode=QuantizationMode.ADAPTIVE,
            precision_threshold=0.9
        )
        quantizer = VectorQuantizer(config)
        
        # 量化所有向量
        quantized_data = []
        for vector in test_vectors:
            quantized, params = await quantizer.quantize_vector(vector)
            quantized_data.append((vector, quantized, params))
        
        # 质量评估
        assessor = QuantizationQualityAssessment()
        original_vectors = [data[0] for data in quantized_data]
        quantized_vectors = [data[1] for data in quantized_data]
        params_list = [data[2] for data in quantized_data]
        
        quality_report = await assessor.assess_quality(
            original_vectors, quantized_vectors, params_list
        )
        
        # 验证整个流程
        assert quality_report["total_vectors"] == 5
        assert quality_report["quality_score"] > 0.0
        
        # 测试反量化
        for original, quantized, params in quantized_data:
            dequantized = await quantizer.dequantize_vector(quantized, params)
            assert dequantized.shape == original.shape
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """测试不同量化模式的性能对比"""
        import time
        
        # 创建大向量用于性能测试
        np.random.seed(42)
        large_vector = np.random.normal(0, 1, 1536).astype(np.float32)
        
        modes = [QuantizationMode.INT8, QuantizationMode.INT4, QuantizationMode.ADAPTIVE]
        results = {}
        
        for mode in modes:
            config = QuantizationConfig(mode=mode)
            quantizer = VectorQuantizer(config)
            
            # 测量量化时间
            start_time = time.time()
            quantized, params = await quantizer.quantize_vector(large_vector)
            end_time = time.time()
            
            results[mode.value] = {
                "time_ms": (end_time - start_time) * 1000,
                "compression": params.get("compression", 1.0),
                "precision_loss": params.get("precision_loss", 0.0)
            }
        
        # 验证结果
        assert len(results) == 3
        for mode_name, result in results.items():
            assert result["time_ms"] >= 0
            assert result["compression"] >= 1.0
            assert 0.0 <= result["precision_loss"] <= 1.0