"""
模型注册表测试

测试PyTorch、ONNX和HuggingFace模型的注册、加载、保存等功能
"""

import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ai.model_service.model_registry import (
    ModelRegistry, ModelMetadata, ModelEntry, ModelFormat, ModelType, CompressionType,
    PyTorchLoader, ONNXLoader, HuggingFaceLoader,
    register_pytorch_model, register_onnx_model, register_huggingface_model,
    model_registry
)


class TestModelMetadata:
    """测试模型元数据类"""
    
    def test_model_metadata_creation(self):
        """测试元数据创建"""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            model_type=ModelType.CLASSIFICATION,
            description="测试模型",
            parameters_count=1000000,
            model_size_mb=100.5
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.format == ModelFormat.PYTORCH
        assert metadata.model_type == ModelType.CLASSIFICATION
        assert metadata.parameters_count == 1000000
        assert metadata.model_size_mb == 100.5
        
        # 测试默认值
        assert metadata.compression_type == CompressionType.NONE
        assert metadata.tags == []
        assert metadata.dependencies == []
    
    def test_metadata_serialization(self):
        """测试元数据序列化"""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            format=ModelFormat.ONNX,
            model_type=ModelType.GENERATION
        )
        
        # 转换为字典
        data = metadata.to_dict()
        assert data['name'] == "test_model"
        assert data['format'] == "onnx"
        assert data['model_type'] == "generation"
        
        # 从字典恢复
        restored = ModelMetadata.from_dict(data)
        assert restored.name == metadata.name
        assert restored.format == metadata.format
        assert restored.model_type == metadata.model_type
    
    def test_update_timestamp(self):
        """测试时间戳更新"""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            model_type=ModelType.CUSTOM
        )
        
        original_time = metadata.updated_at
        metadata.update_timestamp()
        assert metadata.updated_at > original_time


class TestModelEntry:
    """测试模型条目类"""
    
    def test_model_entry_creation(self):
        """测试模型条目创建"""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            model_type=ModelType.CLASSIFICATION
        )
        
        entry = ModelEntry(
            metadata=metadata,
            model_path="/path/to/model.pth",
            config_path="/path/to/config.json"
        )
        
        assert entry.metadata == metadata
        assert entry.model_path == "/path/to/model.pth"
        assert entry.config_path == "/path/to/config.json"
    
    def test_checksum_calculation(self):
        """测试校验和计算"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            metadata = ModelMetadata(
                name="test_model",
                version="1.0.0",
                format=ModelFormat.PYTORCH,
                model_type=ModelType.CUSTOM
            )
            
            entry = ModelEntry(
                metadata=metadata,
                model_path=temp_path
            )
            
            checksum = entry.calculate_checksum()
            assert checksum is not None
            assert len(checksum) == 64  # SHA256长度
            assert entry.checksum == checksum
            
            # 验证完整性
            assert entry.verify_integrity() is True
            
            # 修改文件后验证应该失败
            with open(temp_path, 'w') as f:
                f.write("modified content")
            
            assert entry.verify_integrity() is False
            
        finally:
            os.unlink(temp_path)


class TestPyTorchLoader:
    """测试PyTorch加载器"""
    
    @pytest.fixture
    def pytorch_loader(self):
        """PyTorch加载器fixture"""
        return PyTorchLoader()
    
    @pytest.fixture
    def mock_pytorch_model(self):
        """Mock PyTorch模型"""
        with patch('src.ai.model_service.model_registry.HAS_PYTORCH', True):
            mock_model = Mock()
            mock_model.__class__.__name__ = 'TestModel'
            mock_model.parameters.return_value = [Mock(numel=lambda: 1000), Mock(numel=lambda: 2000)]
            mock_model.training = False
            return mock_model
    
    def test_supported_formats(self, pytorch_loader):
        """测试支持的格式"""
        formats = pytorch_loader.supported_formats()
        assert ModelFormat.PYTORCH in formats
        assert ModelFormat.PYTORCH_SCRIPT in formats
    
    @patch('src.ai.model_service.model_registry.HAS_PYTORCH', True)
    @patch('src.ai.model_service.model_registry.torch')
    def test_load_pytorch_model(self, mock_torch, pytorch_loader):
        """测试PyTorch模型加载"""
        mock_torch.load.return_value = {"state_dict": "test"}
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            result = pytorch_loader.load(f.name, weights_only=True)
            mock_torch.load.assert_called_once()
            assert result == {"state_dict": "test"}
    
    @patch('src.ai.model_service.model_registry.HAS_PYTORCH', True)
    @patch('src.ai.model_service.model_registry.torch')
    @patch('src.ai.model_service.model_registry.nn')
    def test_save_pytorch_model(self, mock_nn, mock_torch, pytorch_loader, mock_pytorch_model):
        """测试PyTorch模型保存"""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            # 模拟nn.Module
            mock_nn.Module = Mock
            mock_pytorch_model.__class__.__bases__ = (mock_nn.Module,)
            mock_pytorch_model.state_dict.return_value = {"param": "value"}
            
            pytorch_loader.save(mock_pytorch_model, temp_path)
            mock_torch.save.assert_called_once()
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.ai.model_service.model_registry.HAS_PYTORCH', True)
    @patch('src.ai.model_service.model_registry.torch')
    def test_get_metadata(self, mock_torch, pytorch_loader, mock_pytorch_model):
        """测试PyTorch模型元数据提取"""
        mock_torch.__version__ = "2.0.0"
        
        metadata = pytorch_loader.get_metadata(mock_pytorch_model)
        
        assert metadata['framework'] == 'pytorch'
        assert metadata['framework_version'] == '2.0.0'
        assert metadata['total_parameters'] == 3000  # 1000 + 2000
        assert metadata['model_class'] == 'TestModel'


class TestONNXLoader:
    """测试ONNX加载器"""
    
    @pytest.fixture
    def onnx_loader(self):
        """ONNX加载器fixture"""
        return ONNXLoader()
    
    @pytest.fixture
    def mock_onnx_model(self):
        """Mock ONNX模型"""
        mock_model = Mock()
        mock_model.ir_version = 7
        mock_model.producer_name = "test_producer"
        mock_model.producer_version = "1.0"
        mock_model.domain = "test.domain"
        mock_model.model_version = 1
        mock_model.doc_string = "Test ONNX model"
        
        # Mock opset imports
        mock_opset = Mock()
        mock_opset.domain = ""
        mock_opset.version = 11
        mock_model.opset_import = [mock_opset]
        
        # Mock graph
        mock_graph = Mock()
        mock_model.graph = mock_graph
        mock_graph.input = []
        mock_graph.output = []
        
        # Mock metadata props
        mock_model.metadata_props = []
        
        return mock_model
    
    def test_supported_formats(self, onnx_loader):
        """测试支持的格式"""
        formats = onnx_loader.supported_formats()
        assert ModelFormat.ONNX in formats
    
    @patch('src.ai.model_service.model_registry.HAS_ONNX', True)
    @patch('src.ai.model_service.model_registry.onnx')
    def test_load_onnx_model(self, mock_onnx, onnx_loader, mock_onnx_model):
        """测试ONNX模型加载"""
        mock_onnx.load.return_value = mock_onnx_model
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            result = onnx_loader.load(f.name)
            mock_onnx.load.assert_called_once()
            assert result == mock_onnx_model
    
    @patch('src.ai.model_service.model_registry.HAS_ONNX', True)
    @patch('src.ai.model_service.model_registry.onnx')
    def test_save_onnx_model(self, mock_onnx, onnx_loader, mock_onnx_model):
        """测试ONNX模型保存"""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name
        
        try:
            onnx_loader.save(mock_onnx_model, temp_path)
            mock_onnx.save.assert_called_once_with(mock_onnx_model, temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.ai.model_service.model_registry.HAS_ONNX', True)
    def test_get_metadata(self, onnx_loader, mock_onnx_model):
        """测试ONNX模型元数据提取"""
        metadata = onnx_loader.get_metadata(mock_onnx_model)
        
        assert metadata['framework'] == 'onnx'
        assert metadata['ir_version'] == 7
        assert metadata['producer_name'] == 'test_producer'
        assert metadata['producer_version'] == '1.0'
        assert metadata['domain'] == 'test.domain'
        assert len(metadata['opset_imports']) == 1
        assert metadata['opset_imports'][0]['version'] == 11


class TestHuggingFaceLoader:
    """测试HuggingFace加载器"""
    
    @pytest.fixture
    def hf_loader(self):
        """HuggingFace加载器fixture"""
        return HuggingFaceLoader()
    
    @pytest.fixture
    def mock_hf_model(self):
        """Mock HuggingFace模型"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "bert"
        mock_model.config.architectures = ["BertModel"]
        mock_model.config.vocab_size = 30522
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.to_dict.return_value = {"model_type": "bert"}
        
        # Mock parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000
        mock_param2.requires_grad = False
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        return mock_model
    
    @pytest.fixture
    def mock_hf_tokenizer(self):
        """Mock HuggingFace tokenizer"""
        mock_tokenizer = Mock()
        return mock_tokenizer
    
    def test_supported_formats(self, hf_loader):
        """测试支持的格式"""
        formats = hf_loader.supported_formats()
        assert ModelFormat.HUGGINGFACE in formats
        assert ModelFormat.SAFETENSORS in formats
    
    @patch('src.ai.model_service.model_registry.HAS_TRANSFORMERS', True)
    @patch('src.ai.model_service.model_registry.AutoModel')
    @patch('src.ai.model_service.model_registry.AutoTokenizer')
    def test_load_hf_model(self, mock_tokenizer_class, mock_model_class, 
                          hf_loader, mock_hf_model, mock_hf_tokenizer):
        """测试HuggingFace模型加载"""
        mock_model_class.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_class.from_pretrained.return_value = mock_hf_tokenizer
        
        model, tokenizer = hf_loader.load("test_model_path")
        
        assert model == mock_hf_model
        assert tokenizer == mock_hf_tokenizer
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
    
    @patch('src.ai.model_service.model_registry.HAS_TRANSFORMERS', True)
    @patch('src.ai.model_service.model_registry.PreTrainedModel')
    @patch('src.ai.model_service.model_registry.PreTrainedTokenizer')
    def test_save_hf_model(self, mock_tokenizer_class, mock_model_class, 
                          hf_loader, mock_hf_model, mock_hf_tokenizer):
        """测试HuggingFace模型保存"""
        # 设置isinstance返回True
        mock_model_class.__instancecheck__ = lambda self, instance: True
        mock_tokenizer_class.__instancecheck__ = lambda self, instance: True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_loader.save(mock_hf_model, temp_dir, tokenizer=mock_hf_tokenizer)
            mock_hf_model.save_pretrained.assert_called_once()
            mock_hf_tokenizer.save_pretrained.assert_called_once()
    
    def test_get_metadata(self, hf_loader, mock_hf_model):
        """测试HuggingFace模型元数据提取"""
        metadata = hf_loader.get_metadata(mock_hf_model)
        
        assert metadata['framework'] == 'huggingface_transformers'
        assert metadata['model_type'] == 'bert'
        assert metadata['vocab_size'] == 30522
        assert metadata['hidden_size'] == 768
        assert metadata['total_parameters'] == 3000


class TestModelRegistry:
    """测试模型注册表"""
    
    @pytest.fixture
    def temp_registry(self):
        """临时模型注册表"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            yield registry
    
    @pytest.fixture
    def mock_simple_model(self):
        """简单的mock模型"""
        return {"weights": [1, 2, 3], "config": {"param": "value"}}
    
    def test_registry_initialization(self, temp_registry):
        """测试注册表初始化"""
        assert isinstance(temp_registry, ModelRegistry)
        assert temp_registry.registry_path.exists()
        assert len(temp_registry.models) == 0
    
    @patch('src.ai.model_service.model_registry.HAS_PYTORCH', True)
    def test_register_model(self, temp_registry, mock_simple_model):
        """测试模型注册"""
        # Mock PyTorch loader
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {
            'total_parameters': 1000,
            'model_size_mb': 10.5,
            'framework_version': '2.0.0'
        }
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        entry = temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            model_type=ModelType.CLASSIFICATION,
            version="1.0.0",
            description="测试模型"
        )
        
        assert isinstance(entry, ModelEntry)
        assert entry.metadata.name == "test_model"
        assert entry.metadata.version == "1.0.0"
        assert entry.metadata.format == ModelFormat.PYTORCH
        assert entry.metadata.model_type == ModelType.CLASSIFICATION
        assert entry.metadata.parameters_count == 1000
        
        # 检查模型是否已注册
        assert "test_model:1.0.0" in temp_registry.models
    
    def test_register_duplicate_model(self, temp_registry, mock_simple_model):
        """测试重复模型注册"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 第一次注册
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            version="1.0.0"
        )
        
        # 第二次注册应该失败
        with pytest.raises(ValueError, match="模型已存在"):
            temp_registry.register_model(
                name="test_model",
                model=mock_simple_model,
                model_format=ModelFormat.PYTORCH,
                version="1.0.0"
            )
        
        # 使用overwrite=True应该成功
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            version="1.0.0",
            overwrite=True
        )
    
    def test_list_models(self, temp_registry, mock_simple_model):
        """测试模型列表"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 注册多个模型
        temp_registry.register_model(
            name="model1",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            model_type=ModelType.CLASSIFICATION
        )
        
        temp_registry.register_model(
            name="model2",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            model_type=ModelType.GENERATION
        )
        
        # 测试列出所有模型
        all_models = temp_registry.list_models()
        assert len(all_models) == 2
        
        # 测试按类型筛选
        classification_models = temp_registry.list_models(ModelType.CLASSIFICATION)
        assert len(classification_models) == 1
        assert classification_models[0].metadata.name == "model1"
    
    def test_get_model_info(self, temp_registry, mock_simple_model):
        """测试获取模型信息"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            version="1.0.0"
        )
        
        # 测试获取特定版本
        info = temp_registry.get_model_info("test_model", "1.0.0")
        assert info is not None
        assert info.metadata.name == "test_model"
        assert info.metadata.version == "1.0.0"
        
        # 测试获取最新版本
        info_latest = temp_registry.get_model_info("test_model", "latest")
        assert info_latest is not None
        assert info_latest.metadata.name == "test_model"
        
        # 测试不存在的模型
        info_none = temp_registry.get_model_info("non_existent")
        assert info_none is None
    
    def test_load_model(self, temp_registry, mock_simple_model):
        """测试模型加载"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        mock_loader.load.return_value = mock_simple_model
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 注册模型
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH
        )
        
        # 加载模型
        loaded_model, tokenizer = temp_registry.load_model("test_model")
        assert loaded_model == mock_simple_model
        assert tokenizer is None  # PyTorch模型没有tokenizer
        
        mock_loader.load.assert_called_once()
    
    def test_remove_model(self, temp_registry, mock_simple_model):
        """测试模型移除"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 注册模型
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH,
            version="1.0.0"
        )
        
        assert "test_model:1.0.0" in temp_registry.models
        
        # 移除特定版本
        result = temp_registry.remove_model("test_model", "1.0.0")
        assert result is True
        assert "test_model:1.0.0" not in temp_registry.models
        
        # 再次移除应该返回False
        result = temp_registry.remove_model("test_model", "1.0.0")
        assert result is False
    
    def test_validate_registry(self, temp_registry, mock_simple_model):
        """测试注册表验证"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 注册一个模型
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH
        )
        
        validation_result = temp_registry.validate_registry()
        
        assert isinstance(validation_result, dict)
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        assert 'total_models' in validation_result
        assert 'valid_models' in validation_result
        
        # 至少应该有一个模型
        assert validation_result['total_models'] >= 1
    
    def test_temporary_model_context(self, temp_registry, mock_simple_model):
        """测试临时模型上下文管理器"""
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        mock_loader.load.return_value = mock_simple_model
        temp_registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        # 注册模型
        temp_registry.register_model(
            name="test_model",
            model=mock_simple_model,
            model_format=ModelFormat.PYTORCH
        )
        
        # 使用上下文管理器
        with temp_registry.temporary_model("test_model") as (model, tokenizer):
            assert model == mock_simple_model
            assert tokenizer is None


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @patch('src.ai.model_service.model_registry.model_registry')
    def test_register_pytorch_model(self, mock_registry):
        """测试PyTorch模型注册便捷函数"""
        mock_model = Mock()
        mock_entry = Mock()
        mock_registry.register_model.return_value = mock_entry
        
        result = register_pytorch_model("test_model", mock_model, version="2.0.0")
        
        mock_registry.register_model.assert_called_once_with(
            name="test_model",
            model=mock_model,
            model_format=ModelFormat.PYTORCH,
            version="2.0.0"
        )
        assert result == mock_entry
    
    @patch('src.ai.model_service.model_registry.model_registry')
    def test_register_onnx_model(self, mock_registry):
        """测试ONNX模型注册便捷函数"""
        mock_model = Mock()
        mock_entry = Mock()
        mock_registry.register_model.return_value = mock_entry
        
        result = register_onnx_model("test_model", mock_model)
        
        mock_registry.register_model.assert_called_once_with(
            name="test_model",
            model=mock_model,
            model_format=ModelFormat.ONNX,
            version="1.0.0"
        )
        assert result == mock_entry
    
    @patch('src.ai.model_service.model_registry.model_registry')
    def test_register_huggingface_model(self, mock_registry):
        """测试HuggingFace模型注册便捷函数"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_entry = Mock()
        mock_registry.register_model.return_value = mock_entry
        
        result = register_huggingface_model("test_model", mock_model, mock_tokenizer)
        
        mock_registry.register_model.assert_called_once_with(
            name="test_model",
            model=mock_model,
            model_format=ModelFormat.HUGGINGFACE,
            tokenizer=mock_tokenizer,
            version="1.0.0"
        )
        assert result == mock_entry


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def integration_registry(self):
        """集成测试的注册表"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ModelRegistry(temp_dir)
    
    @patch('src.ai.model_service.model_registry.HAS_PYTORCH', True)
    @patch('src.ai.model_service.model_registry.torch')
    def test_full_pytorch_workflow(self, mock_torch, integration_registry):
        """测试完整的PyTorch工作流程"""
        # 准备mock对象
        mock_model = Mock()
        mock_model.__class__.__name__ = 'TestModel'
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_model.training = False
        mock_model.state_dict.return_value = {"param": "value"}
        
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            mock_model_path = f.name
        
        try:
            # 注册模型
            entry = integration_registry.register_model(
                name="integration_model",
                model=mock_model,
                model_format=ModelFormat.PYTORCH,
                model_type=ModelType.CLASSIFICATION,
                description="集成测试模型"
            )
            
            assert entry is not None
            assert entry.metadata.name == "integration_model"
            
            # 列出模型
            models = integration_registry.list_models()
            assert len(models) == 1
            
            # 获取模型信息
            info = integration_registry.get_model_info("integration_model")
            assert info is not None
            
            # 验证注册表
            validation = integration_registry.validate_registry()
            assert validation['total_models'] == 1
            
        finally:
            if os.path.exists(mock_model_path):
                os.unlink(mock_model_path)
    
    def test_registry_persistence(self):
        """测试注册表持久化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建注册表并注册模型
            registry1 = ModelRegistry(temp_dir)
            
            mock_loader = Mock()
            mock_loader.save = Mock()
            mock_loader.get_metadata.return_value = {}
            registry1.loaders[ModelFormat.PYTORCH] = mock_loader
            
            registry1.register_model(
                name="persistent_model",
                model={"data": "test"},
                model_format=ModelFormat.PYTORCH
            )
            
            assert len(registry1.models) == 1
            
            # 创建新的注册表实例，应该能加载已保存的模型
            registry2 = ModelRegistry(temp_dir)
            assert len(registry2.models) == 1
            assert "persistent_model:1.0.0" in registry2.models
            
            # 验证模型信息
            info = registry2.get_model_info("persistent_model")
            assert info is not None
            assert info.metadata.name == "persistent_model"


if __name__ == "__main__":
    pytest.main([__file__])