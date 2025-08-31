"""模型注册表单元测试"""

import asyncio
import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock

from ....ai.model_service.registry import (
    ModelRegistry,
    ModelRegistrationRequest,
    ModelFormat,
    ModelMetadata,
    PyTorchLoader,
    ONNXLoader,
    HuggingFaceLoader
)

class TestModelRegistry:
    """模型注册表测试"""
    
    @pytest.fixture
    def temp_storage(self):
        """临时存储目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def model_registry(self, temp_storage):
        """模型注册表实例"""
        return ModelRegistry(temp_storage)
    
    @pytest.fixture
    def simple_model(self):
        """简单PyTorch模型"""
        return nn.Linear(10, 5)
    
    @pytest.mark.asyncio
    async def test_register_pytorch_model(self, model_registry, simple_model, temp_storage):
        """测试注册PyTorch模型"""
        # 保存模型文件
        model_path = Path(temp_storage) / "test_model.pt"
        torch.save(simple_model, model_path)
        
        # 创建注册请求
        request = ModelRegistrationRequest(
            name="test-pytorch-model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch",
            description="测试PyTorch模型",
            tags=["test", "pytorch"]
        )
        
        # 注册模型
        model_id = await model_registry.register_model(request, str(model_path))
        
        # 验证注册成功
        assert model_id is not None
        
        # 验证模型信息
        model_info = model_registry.get_model("test-pytorch-model", "1.0.0")
        assert model_info is not None
        assert model_info.name == "test-pytorch-model"
        assert model_info.version == "1.0.0"
        assert model_info.format == ModelFormat.PYTORCH
        assert model_info.framework == "pytorch"
        assert model_info.description == "测试PyTorch模型"
        assert "test" in model_info.tags
        assert "pytorch" in model_info.tags
    
    @pytest.mark.asyncio
    async def test_register_model_with_metadata_extraction(self, model_registry, simple_model, temp_storage):
        """测试注册模型并提取元数据"""
        model_path = Path(temp_storage) / "metadata_model.pt"
        torch.save(simple_model, model_path)
        
        request = ModelRegistrationRequest(
            name="metadata-model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        model_id = await model_registry.register_model(request, str(model_path))
        model_info = model_registry.get_model("metadata-model", "1.0.0")
        
        # 验证元数据提取
        assert model_info.parameter_count is not None
        assert model_info.parameter_count > 0  # Linear(10, 5) 应该有参数
        assert model_info.model_size_mb is not None
        assert model_info.model_size_mb > 0
        assert model_info.checksum is not None
    
    def test_get_model_latest_version(self, model_registry):
        """测试获取最新版本模型"""
        # 手动添加多个版本
        from datetime import datetime, timezone
        
        v1_metadata = ModelMetadata(
            model_id="test-1",
            name="version-test",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch",
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc)
        )
        
        v2_metadata = ModelMetadata(
            model_id="test-2", 
            name="version-test",
            version="2.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch",
            created_at=datetime(2023, 2, 1, tzinfo=timezone.utc)
        )
        
        # 添加到注册表
        if "version-test" not in model_registry.models:
            model_registry.models["version-test"] = {}
        
        model_registry.models["version-test"]["1.0.0"] = v1_metadata
        model_registry.models["version-test"]["2.0.0"] = v2_metadata
        
        # 测试获取最新版本
        latest = model_registry.get_model("version-test", "latest")
        assert latest is not None
        assert latest.version == "2.0.0"  # 应该返回最新的版本
    
    def test_list_models_with_filters(self, model_registry):
        """测试带过滤条件的模型列表"""
        # 添加测试数据
        models = [
            ModelMetadata(
                model_id="test-1",
                name="bert-model",
                version="1.0.0",
                format=ModelFormat.HUGGINGFACE,
                framework="transformers",
                tags=["nlp", "bert"]
            ),
            ModelMetadata(
                model_id="test-2",
                name="resnet-model", 
                version="1.0.0",
                format=ModelFormat.PYTORCH,
                framework="pytorch",
                tags=["vision", "cnn"]
            ),
            ModelMetadata(
                model_id="test-3",
                name="bert-large",
                version="1.0.0", 
                format=ModelFormat.ONNX,
                framework="onnx",
                tags=["nlp", "bert", "large"]
            )
        ]
        
        # 添加到注册表
        for model in models:
            if model.name not in model_registry.models:
                model_registry.models[model.name] = {}
            model_registry.models[model.name][model.version] = model
        
        # 测试名称过滤
        bert_models = model_registry.list_models(name_filter="bert")
        assert len(bert_models) == 2
        
        # 测试标签过滤
        nlp_models = model_registry.list_models(tags=["nlp"])
        assert len(nlp_models) == 2
        
        # 测试组合过滤
        bert_nlp_models = model_registry.list_models(name_filter="bert", tags=["nlp"])
        assert len(bert_nlp_models) == 2
    
    @pytest.mark.asyncio
    async def test_delete_model(self, model_registry, simple_model, temp_storage):
        """测试删除模型"""
        # 先注册一个模型
        model_path = Path(temp_storage) / "delete_test.pt"
        torch.save(simple_model, model_path)
        
        request = ModelRegistrationRequest(
            name="delete-test",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        model_id = await model_registry.register_model(request, str(model_path))
        
        # 验证模型存在
        assert model_registry.get_model("delete-test", "1.0.0") is not None
        
        # 删除模型
        success = model_registry.delete_model("delete-test", "1.0.0")
        assert success
        
        # 验证模型已删除
        assert model_registry.get_model("delete-test", "1.0.0") is None
    
    def test_validate_model_integrity(self, model_registry, simple_model, temp_storage):
        """测试模型完整性验证"""
        # 创建模型文件
        model_path = Path(temp_storage) / "validate_test.pt"
        torch.save(simple_model, model_path)
        
        # 手动创建模型元数据（模拟已注册的模型）
        import hashlib
        with open(model_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()
        
        metadata = ModelMetadata(
            model_id="validate-test",
            name="validate-model",
            version="1.0.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch",
            checksum=checksum
        )
        
        # 添加到注册表
        model_registry.models["validate-model"] = {"1.0.0": metadata}
        
        # 模拟模型文件路径
        model_registry.get_model_path = lambda name, version: str(model_path)
        
        # 验证模型
        result = model_registry.validate_model("validate-model", "1.0.0")
        assert result["valid"] is True
        assert "message" in result
    
    def test_get_statistics(self, model_registry):
        """测试获取统计信息"""
        # 添加测试数据
        models = [
            ModelMetadata(
                model_id="stats-1",
                name="model1",
                version="1.0.0", 
                format=ModelFormat.PYTORCH,
                framework="pytorch",
                model_size_mb=100.5
            ),
            ModelMetadata(
                model_id="stats-2",
                name="model2",
                version="1.0.0",
                format=ModelFormat.ONNX,
                framework="onnx",
                model_size_mb=50.2
            ),
            ModelMetadata(
                model_id="stats-3", 
                name="model1",
                version="2.0.0",
                format=ModelFormat.PYTORCH,
                framework="pytorch",
                model_size_mb=120.0
            )
        ]
        
        # 添加到注册表
        for model in models:
            if model.name not in model_registry.models:
                model_registry.models[model.name] = {}
            model_registry.models[model.name][model.version] = model
        
        stats = model_registry.get_statistics()
        
        assert stats["total_model_families"] == 2  # model1, model2
        assert stats["total_model_versions"] == 3
        assert stats["formats"]["pytorch"] == 2
        assert stats["formats"]["onnx"] == 1
        assert stats["frameworks"]["pytorch"] == 2
        assert stats["frameworks"]["onnx"] == 1
        assert abs(stats["total_storage_mb"] - 270.7) < 0.1

class TestModelLoaders:
    """模型加载器测试"""
    
    @pytest.fixture
    def simple_pytorch_model(self):
        """简单PyTorch模型"""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    def test_pytorch_loader_extract_metadata(self, simple_pytorch_model, tmp_path):
        """测试PyTorch模型元数据提取"""
        model_path = tmp_path / "pytorch_test.pt"
        torch.save(simple_pytorch_model, model_path)
        
        metadata = PyTorchLoader.extract_metadata(str(model_path))
        
        # framework字段已从extract_metadata返回值中移除
        assert "parameter_count" in metadata
        assert "model_size_mb" in metadata
        assert metadata["parameter_count"] > 0  # 应该有参数
        assert metadata["model_size_mb"] > 0
    
    def test_pytorch_loader_safe_loading(self, simple_pytorch_model, tmp_path):
        """测试PyTorch安全加载"""
        model_path = tmp_path / "safe_load_test.pt"
        torch.save(simple_pytorch_model, model_path)
        
        # 测试加载模型
        loaded_model = PyTorchLoader.load_model(str(model_path), "cpu")
        assert loaded_model is not None
        
        # 测试模型结构一致性
        assert len(list(loaded_model.parameters())) == len(list(simple_pytorch_model.parameters()))
    
    @patch('onnx.load')
    def test_onnx_loader_extract_metadata(self, mock_onnx_load, tmp_path):
        """测试ONNX模型元数据提取"""
        # 模拟ONNX模型
        mock_model = MagicMock()
        mock_model.graph.input = [MagicMock()]
        mock_model.graph.output = [MagicMock()]
        mock_model.opset_import = [MagicMock()]
        mock_model.opset_import[0].version = 11
        
        # 配置输入输出信息
        mock_input = mock_model.graph.input[0]
        mock_input.name = "input"
        mock_input.type.tensor_type.elem_type = 1  # FLOAT
        mock_input.type.tensor_type.shape.dim = [MagicMock(), MagicMock()]
        mock_input.type.tensor_type.shape.dim[0].dim_value = 1
        mock_input.type.tensor_type.shape.dim[1].dim_value = 10
        
        mock_output = mock_model.graph.output[0]
        mock_output.name = "output"
        mock_output.type.tensor_type.elem_type = 1
        mock_output.type.tensor_type.shape.dim = [MagicMock()]
        mock_output.type.tensor_type.shape.dim[0].dim_value = 1
        
        mock_onnx_load.return_value = mock_model
        
        # 创建临时文件
        onnx_path = tmp_path / "test.onnx"
        onnx_path.write_bytes(b"fake onnx data")
        
        metadata = ONNXLoader.extract_metadata(str(onnx_path))
        
        # framework字段已从extract_metadata返回值中移除
        assert "input_schema" in metadata
        assert "output_schema" in metadata
        assert metadata["opset_version"] == 11
    
    @patch('transformers.AutoConfig.from_pretrained')
    def test_huggingface_loader_extract_metadata(self, mock_config):
        """测试HuggingFace模型元数据提取"""
        # 模拟配置
        mock_config_instance = MagicMock()
        mock_config_instance.model_type = "bert"
        mock_config_instance.vocab_size = 30522
        mock_config_instance.hidden_size = 768
        mock_config_instance.num_hidden_layers = 12
        mock_config_instance.num_attention_heads = 12
        mock_config.return_value = mock_config_instance
        
        metadata = HuggingFaceLoader.extract_metadata("bert-base-uncased")
        
        # 不再检查framework字段，因为已从返回的metadata中移除
        assert metadata["model_type"] == "bert"
        assert metadata["vocab_size"] == 30522
        assert metadata["hidden_size"] == 768
        assert metadata["num_layers"] == 12
        assert metadata["num_attention_heads"] == 12
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_huggingface_loader_load_model(self, mock_config, mock_tokenizer, mock_model):
        """测试HuggingFace模型加载"""
        # 配置mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_config_instance = MagicMock()
        
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_config.return_value = mock_config_instance
        
        # 测试加载
        model, tokenizer, config = HuggingFaceLoader.load_model_and_tokenizer("bert-base-uncased")
        
        assert model is mock_model_instance
        assert tokenizer is mock_tokenizer_instance
        assert config is mock_config_instance
        
        # 验证调用
        mock_model.assert_called_once_with("bert-base-uncased")
        mock_tokenizer.assert_called_once_with("bert-base-uncased")
        mock_config.assert_called_once_with("bert-base-uncased")

if __name__ == "__main__":
    import tempfile
    
    # 手动运行测试示例
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(temp_dir)
        
        # 测试基本功能
        simple_model = nn.Linear(10, 1)
        model_path = Path(temp_dir) / "manual_test.pt" 
        torch.save(simple_model, model_path)
        
        async def manual_test():
            request = ModelRegistrationRequest(
                name="manual-test",
                version="1.0.0",
                format=ModelFormat.PYTORCH,
                framework="pytorch"
            )
            
            model_id = await registry.register_model(request, str(model_path))
            print(f"注册成功，模型ID: {model_id}")
            
            model_info = registry.get_model("manual-test", "1.0.0")
            print(f"模型信息: {model_info}")
            
            stats = registry.get_statistics()
            print(f"统计信息: {stats}")
        
        asyncio.run(manual_test())
        print("手动测试完成")