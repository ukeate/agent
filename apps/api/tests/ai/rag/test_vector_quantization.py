"""
向量量化模块测试
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from src.ai.rag.quantization import (
    BinaryQuantizer, 
    HalfPrecisionQuantizer, 
    QuantizationManager
)


class TestBinaryQuantizer:
    """测试二进制量化器"""
    
    def test_initialization(self):
        """测试初始化"""
        quantizer = BinaryQuantizer(bits=8)
        
        assert quantizer.name == "binary"
        assert quantizer.bits == 8
        assert quantizer.is_trained is False
        assert quantizer.thresholds is None
    
    def test_train_success(self):
        """测试训练成功"""
        quantizer = BinaryQuantizer()
        
        # 创建测试向量
        test_vectors = np.array([
            [0.1, 0.5, -0.2],
            [0.3, -0.1, 0.8],  
            [-0.2, 0.2, 0.1],
            [0.0, -0.3, 0.4]
        ])
        
        result = quantizer.train(test_vectors)
        
        assert result is True
        assert quantizer.is_trained is True
        assert quantizer.thresholds is not None
        assert len(quantizer.thresholds) == 3
    
    def test_encode_decode_cycle(self):
        """测试编码-解码循环"""
        quantizer = BinaryQuantizer()
        
        # 训练数据
        training_vectors = np.array([
            [0.5, -0.5, 0.0],
            [1.0, -1.0, 0.5],
            [-0.5, 0.5, -0.5]
        ])
        
        quantizer.train(training_vectors)
        
        # 测试向量
        test_vectors = np.array([
            [0.7, -0.3, 0.2],
            [-0.2, 0.8, -0.1]
        ])
        
        # 编码
        encoded = quantizer.encode(test_vectors)
        
        assert encoded.dtype == np.uint8
        assert encoded.shape == test_vectors.shape
        
        # 解码
        decoded = quantizer.decode(encoded)
        
        assert decoded.shape == test_vectors.shape
        assert decoded.dtype == np.float64
    
    def test_encode_without_training(self):
        """测试未训练时编码"""
        quantizer = BinaryQuantizer()
        
        test_vectors = np.array([[0.1, 0.2, 0.3]])
        
        with pytest.raises(ValueError, match="量化器未训练"):
            quantizer.encode(test_vectors)
    
    def test_decode_without_training(self):
        """测试未训练时解码"""
        quantizer = BinaryQuantizer()
        
        test_codes = np.array([[1, 0, 1]], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="量化器未训练"):
            quantizer.decode(test_codes)
    
    def test_get_params(self):
        """测试获取参数"""
        quantizer = BinaryQuantizer(bits=4)
        
        # 未训练时
        params = quantizer.get_params()
        
        assert params["type"] == "binary"
        assert params["bits"] == 4
        assert params["thresholds"] is None
        assert params["compression_ratio"] == 8.0  # 32/4
        
        # 训练后
        training_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        quantizer.train(training_data)
        
        params = quantizer.get_params()
        assert params["thresholds"] is not None
        assert len(params["thresholds"]) == 2


class TestHalfPrecisionQuantizer:
    """测试半精度量化器"""
    
    def test_initialization(self):
        """测试初始化"""
        quantizer = HalfPrecisionQuantizer()
        
        assert quantizer.name == "halfprecision"
        assert quantizer.is_trained is False
    
    def test_train(self):
        """测试训练（半精度不需要训练）"""
        quantizer = HalfPrecisionQuantizer()
        
        # 任意向量，半精度不需要训练
        vectors = np.array([[0.1, 0.2, 0.3]])
        
        result = quantizer.train(vectors)
        
        assert result is True
        assert quantizer.is_trained is True
    
    def test_encode_decode_cycle(self):
        """测试编码-解码循环"""
        quantizer = HalfPrecisionQuantizer()
        quantizer.train(np.array([[0.0]]))  # 形式上的训练
        
        # 测试向量
        test_vectors = np.array([
            [0.123456789, -0.987654321, 0.5],
            [1.234567890, -2.345678901, -0.5]
        ], dtype=np.float32)
        
        # 编码
        encoded = quantizer.encode(test_vectors)
        
        assert encoded.dtype == np.float16
        assert encoded.shape == test_vectors.shape
        
        # 解码
        decoded = quantizer.decode(encoded)
        
        assert decoded.dtype == np.float32
        assert decoded.shape == test_vectors.shape
        
        # 检查精度损失在可接受范围内
        assert np.allclose(decoded, test_vectors, rtol=1e-3)
    
    def test_get_params(self):
        """测试获取参数"""
        quantizer = HalfPrecisionQuantizer()
        
        params = quantizer.get_params()
        
        assert params["type"] == "halfprecision"
        assert params["compression_ratio"] == 2.0


class TestQuantizationManager:
    """测试量化管理器"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """模拟向量存储"""
        mock_store = MagicMock()
        mock_store.get_connection = AsyncMock()
        return mock_store
    
    @pytest.fixture
    def quantization_manager(self, mock_vector_store):
        """创建量化管理器实例"""
        return QuantizationManager(mock_vector_store)
    
    @pytest.mark.asyncio
    async def test_create_binary_quantization_config(self, quantization_manager):
        """测试创建二进制量化配置"""
        # Mock获取训练向量
        quantization_manager._get_training_vectors = AsyncMock(
            return_value=np.array([
                [0.1, 0.5, -0.2],
                [0.3, -0.1, 0.8],
                [-0.2, 0.2, 0.1]
            ])
        )
        quantization_manager._save_quantization_config = AsyncMock(return_value=True)
        
        result = await quantization_manager.create_quantization_config(
            collection_name="test_collection",
            quantization_type="binary",
            config={"bits": 8, "training_size": 1000}
        )
        
        assert result is True
        assert "test_collection" in quantization_manager.quantizers
        assert isinstance(quantization_manager.quantizers["test_collection"], BinaryQuantizer)
    
    @pytest.mark.asyncio
    async def test_create_halfprecision_quantization_config(self, quantization_manager):
        """测试创建半精度量化配置"""
        # Mock获取训练向量
        quantization_manager._get_training_vectors = AsyncMock(
            return_value=np.array([
                [0.1, 0.5, -0.2],
                [0.3, -0.1, 0.8]
            ])
        )
        quantization_manager._save_quantization_config = AsyncMock(return_value=True)
        
        result = await quantization_manager.create_quantization_config(
            collection_name="test_collection",
            quantization_type="halfprecision"
        )
        
        assert result is True
        assert "test_collection" in quantization_manager.quantizers
        assert isinstance(quantization_manager.quantizers["test_collection"], HalfPrecisionQuantizer)
    
    @pytest.mark.asyncio
    async def test_create_quantization_config_unsupported_type(self, quantization_manager):
        """测试不支持的量化类型"""
        quantization_manager._get_training_vectors = AsyncMock(
            return_value=np.array([[0.1, 0.2]])
        )
        
        with pytest.raises(ValueError, match="不支持的量化类型"):
            await quantization_manager.create_quantization_config(
                collection_name="test_collection",
                quantization_type="unsupported"
            )
    
    @pytest.mark.asyncio
    async def test_create_quantization_config_no_training_data(self, quantization_manager):
        """测试没有训练数据"""
        quantization_manager._get_training_vectors = AsyncMock(return_value=np.array([]))
        
        result = await quantization_manager.create_quantization_config(
            collection_name="empty_collection",
            quantization_type="binary"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_apply_quantization(self, quantization_manager):
        """测试应用量化"""
        # 准备量化器
        quantizer = BinaryQuantizer()
        quantizer.train(np.array([[0.1, 0.5], [0.3, -0.1]]))
        quantization_manager.quantizers["test_collection"] = quantizer
        
        # Mock相关方法
        quantization_manager._get_vectors_batch = AsyncMock(
            return_value=np.array([[0.2, 0.6], [0.4, -0.2]])
        )
        quantization_manager._save_quantized_vectors = AsyncMock(return_value=True)
        quantization_manager._get_collection_vector_count = AsyncMock(return_value=2)
        
        result = await quantization_manager.apply_quantization("test_collection")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_apply_quantization_no_quantizer(self, quantization_manager):
        """测试没有量化器时应用量化"""
        result = await quantization_manager.apply_quantization("nonexistent_collection")
        
        assert result is False
    
    def test_quantization_performance_comparison(self):
        """测试量化性能比较"""
        # 创建测试数据
        original_vectors = np.random.random((1000, 256)).astype(np.float32)
        
        # 二进制量化
        binary_quantizer = BinaryQuantizer()
        binary_quantizer.train(original_vectors)
        binary_encoded = binary_quantizer.encode(original_vectors)
        
        # 半精度量化
        half_quantizer = HalfPrecisionQuantizer() 
        half_quantizer.train(original_vectors)
        half_encoded = half_quantizer.encode(original_vectors)
        
        # 检查压缩效果
        original_size = original_vectors.nbytes
        binary_size = binary_encoded.nbytes
        half_size = half_encoded.nbytes
        
        # 二进制量化应该显著减少存储空间
        assert binary_size < original_size / 4  # 至少4倍压缩
        
        # 半精度量化应该减少一半存储空间
        assert half_size == original_size // 2
        
        print(f"原始大小: {original_size} bytes")
        print(f"二进制量化: {binary_size} bytes (压缩比: {original_size/binary_size:.2f})")
        print(f"半精度量化: {half_size} bytes (压缩比: {original_size/half_size:.2f})")


@pytest.mark.parametrize("quantization_type,expected_quantizer", [
    ("binary", BinaryQuantizer),
    ("halfprecision", HalfPrecisionQuantizer),
])
def test_quantizer_factory(quantization_type, expected_quantizer):
    """参数化测试量化器工厂模式"""
    if quantization_type == "binary":
        quantizer = BinaryQuantizer()
    elif quantization_type == "halfprecision":
        quantizer = HalfPrecisionQuantizer()
    else:
        pytest.fail(f"未知的量化类型: {quantization_type}")
    
    assert isinstance(quantizer, expected_quantizer)
    assert quantizer.name == quantization_type